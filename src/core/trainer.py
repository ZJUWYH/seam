import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
import wandb
from transformers import Trainer
from transformers.trainer import *


class SeamTrainer(Trainer):
    def __init__(self, seam_args=None, align_dataset: Optional[Union[Dataset, IterableDataset]] = None,benign_dataset: Optional[Union[Dataset, IterableDataset]] = None, *args, **kwargs):
        self.seam_args = seam_args  # custom args for training seam model
        self.align_dataset = align_dataset # dataset of harmful prompt - refusal response
        self.benign_dataset = benign_dataset # dataset of benign prompt - benign response
        self.grad_device1, self.grad_device2 = self.gpu_id[1] # Determain GPUs used for extra gradient transfer
        self.model_device_list = self.gpu_id[0]  # Determine GPUs used for data parallel
        self.total_grad = {} # store the gradient for gradient accumulation
        super().__init__(*args, **kwargs)


    @property
    def gpu_id(self):
        """
        Returns the GPU ids used for data paralleling (except for last two gpus) and extra gradient transfer (last two gpus).
        Note:
            if smaller base models are adopted, we can use 1 GPU for extra gradient transfer. And you can revise the code below.
        """
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            device_list = visible_devices.split(',')
            if len(device_list) >= 3:
                first_gpu_id_list = list(range(len(device_list)-2))
                last_two_gpus = [f'cuda:{len(device_list)-2}', f'cuda:{len(device_list)-1}']
                return first_gpu_id_list, last_two_gpus
            else:
                print("Available GPU less than three")
        else:
            num_gpus = torch.cuda.device_count()
            if num_gpus >= 3:
                first_gpu_id_list = list(range(num_gpus-2))
                last_two_gpus = [f'cuda:{num_gpus-2}', f'cuda:{num_gpus-1}']
                return first_gpu_id_list, last_two_gpus
            else:
                print("Available GPU less than three")

    def _wrap_model(self, model, training=True, dataloader=None):
        """
        Use GPUs except for last two GPUs for data parallel instead of all GPUs by default.
        """
        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model) is not model:
            return model

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model, self.model_device_list) # use all GPUs except the last two GPUs

        return model

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        Only difference is we use self.total_grad for gradient accumulation instead of store the gradient in Model instance since we need to load grads for multiple times.
        """
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.state.init_training_references(self, train_dataloader, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                    
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        #################################################
                        for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
                            if param.requires_grad:
                                try:
                                    if param.grad is None:
                                        param.grad = self.total_grad[name].to(param.data.device)
                                    else:
                                        param.grad.copy_(self.total_grad[name].to(param.data.device))
                                    param.grad /= args.gradient_accumulation_steps
                                except Exception as e:
                                    print(f"Error occurred: {e}")
                        self.total_grad.clear()
                        #############################################

                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
        



    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Customized to support multiple datasets for training.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        align_dataset = self.align_dataset
        benign_dataset = self.benign_dataset


        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": False,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return list(zip(
            self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params)),
            self.accelerator.prepare(DataLoader(align_dataset, **dataloader_params)),
            self.accelerator.prepare(DataLoader(benign_dataset, **dataloader_params))
        ))

    def _prepare_inputs(self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], Dict[str, Union[torch.Tensor, Any]]]) -> Tuple[Dict, Dict]:
        """
        Customized to support multiple datasets for training.
        """
        def _prepare(inputs):
            processed = self._prepare_input(inputs)
            if len(processed) == 0:
                raise ValueError("Received empty batch after preparation")
            if self.args.past_index >= 0 and self._past is not None:
                processed["mems"] = self._past
            return processed

        harmful_prepared = _prepare(inputs[0])
        harmless_prepared = _prepare(inputs[1])
        benign_prepared = _prepare(inputs[2])
        return (harmful_prepared, harmless_prepared, benign_prepared)


    @contextmanager
    def temporary_parameter_update(self, model, grad_1: dict, grad_2: dict, 
                                grad_1_norm: torch.Tensor, grad_2_norm: torch.Tensor, 
                                cosine_similarity: torch.Tensor, lr: float):
        """
        Apply the pertubation to the model parameters temporarily for Hessian-free estimation.
        Args:
            model (`nn.Module`):
                The model to train.
            grad_1 (`dict`):
                The gradient of the first model.
            grad_2 (`dict`):
                The gradient of the second model.
            grad_1_norm (`torch.Tensor`):
                The norm of the first gradient.
            grad_2_norm (`torch.Tensor`):
                The norm of the second gradient.
            cosine_similarity (`torch.Tensor`):
                The cosine similarity between the two gradients.
            lr (`float`):
                The learning rate for the update.
        """
        # Store original parameter states
        original_state = {
            name: param.data.clone().to(self.grad_device1)
            for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters())
        }
                
        try:
            model.zero_grad()
            with torch.no_grad():
                # Apply temporary updates based on gradient projection
                for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
                    if param.requires_grad == True:
                        try:
                            param.data.add_((grad_1[name]/grad_1_norm - cosine_similarity * grad_2[name]/grad_2_norm).to(param.data.device), alpha=lr)
                        except:
                            param.data += (lr * (grad_1[name]/grad_1_norm - cosine_similarity * grad_2[name]/grad_2_norm).to(param.data.device))
            yield
            
        finally:
            with torch.no_grad():
                # Restore original parameter values
                for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
                    if param.requires_grad:
                        param.data = original_state[name].to(param.data.device)
                    
                # Clear gradients
                model.zero_grad()
            del original_state

    def get_grad(self, model, device, zero_grad=False):
        # Collect gradients from model parameters
        grad = {}
        for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
            if param.requires_grad:
                if param.grad is not None:
                    grad[name] = param.grad.detach().clone().to(device)
                else:
                    grad[name] = torch.zeros_like(param.data, device=device)
        
        # Optionally clear gradients after collection
        if zero_grad:
            model.zero_grad()
            
        return grad


    def training_step(
        self, model: nn.Module, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], Dict[str, Union[torch.Tensor, Any]], Dict[str, Union[torch.Tensor, Any]]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Tuple[Dict[str, Union[torch.Tensor, Any]], Dict[str, Union[torch.Tensor, Any]], Dict[str, Union[torch.Tensor, Any]]`):
                The inputs to the model. This should be a tuple of three dictionaries, each containing the inputs for
                the harmful, harmless, and benign datasets respectively.
        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
            param.requires_grad = True
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        assert is_sagemaker_mp_enabled() == False, "SageMaker MP is not supported in this method."

        def backward_hf(loss):
            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learnign rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()

            assert self.use_apex == False, "Apex is not supported in this method."
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

        def clear_memory():
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                if is_torch_xpu_available():
                    torch.xpu.empty_cache()
                elif is_torch_mlu_available():
                    torch.mlu.empty_cache()
                elif is_torch_musa_available():
                    torch.musa.empty_cache()
                elif is_torch_npu_available():
                    torch.npu.empty_cache()
                elif is_torch_mps_available(min_version="2.0"):
                    torch.mps.empty_cache()
                else:
                    torch.cuda.empty_cache()
        ##################################################
        # Gradient of loss_ul
        with self.compute_loss_context_manager():
            loss_harmful = self.compute_loss(model, inputs, split = "harmful", num_items_in_batch=num_items_in_batch)
        clear_memory()
        loss_harmful = backward_hf(loss_harmful)
        harmful_grad  = self.get_grad(model = model, device = self.grad_device1,zero_grad=True)

        # Gradient of loss_up
        with self.compute_loss_context_manager():
            loss_harmless = self.compute_loss(model, inputs, split = "harmless", num_items_in_batch=num_items_in_batch)
        clear_memory()
        loss_harmless = backward_hf(loss_harmless)
        harmless_grad = self.get_grad(model = model, device = self.grad_device1,zero_grad=True)

        # calculate the total loss
        output_loss = loss_harmful + self.seam_args.alpha * loss_harmless

        # calculate the total gradient
        with torch.no_grad():
            for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
                if param.requires_grad:
                    if name in self.total_grad.keys():
                        self.total_grad[name] += harmful_grad[name] + self.seam_args.alpha * harmless_grad[name]
                    else:
                        self.total_grad[name] = harmful_grad[name] + self.seam_args.alpha * harmless_grad[name]
            del harmful_grad, harmless_grad

        ###############################################
        # calculated estimated gradient of loss_sd
        # Harmful finetuning gradient
        with self.compute_loss_context_manager():
            loss_harmful = self.compute_loss(model, inputs, split = "attack", num_items_in_batch=num_items_in_batch)
        clear_memory()
        loss_harmful = backward_hf(loss_harmful)
        harmful_grad = self.get_grad(model = model, device = self.grad_device1,zero_grad=True)

        # Benign gradient
        with self.compute_loss_context_manager():
            loss_harmless = self.compute_loss(model, inputs, split = "benign", num_items_in_batch=num_items_in_batch)
        clear_memory()
        loss_harmless = backward_hf(loss_harmless)
        harmless_grad = self.get_grad(model = model, device = self.grad_device1,zero_grad=True)


        harmful_grad_norm = self.compute_l2_norm(harmful_grad)
        wandb.log({"Harmful Gradient Norm": harmful_grad_norm.item()})
        harmless_grad_norm = self.compute_l2_norm(harmless_grad)
        wandb.log({"Harmless Gradient Norm": harmless_grad_norm.item()})
        cosine_similarity = self.compute_cosine_similarity(harmful_grad, harmless_grad)
        wandb.log({"Cosine Similarity": cosine_similarity.item()})

        # Harmful fine-tuning gradient after parameter pertubation
        with self.temporary_parameter_update(model, harmless_grad, harmful_grad, harmless_grad_norm, harmful_grad_norm, cosine_similarity, self.seam_args.epsilon):
            with self.compute_loss_context_manager():
                loss_harmful_p = self.compute_loss(model, inputs, split = "attack", num_items_in_batch=num_items_in_batch)
            clear_memory()
            loss_harmful_p = backward_hf(loss_harmful_p)
            harmful_grad_p  = self.get_grad(model = model, device = self.grad_device2,zero_grad=True)

        # benign gradient after parameter pertubation
        with self.temporary_parameter_update(model, harmful_grad, harmless_grad, harmful_grad_norm, harmless_grad_norm, cosine_similarity, self.seam_args.epsilon):
            with self.compute_loss_context_manager():
                loss_harmless_p = self.compute_loss(model, inputs, split = "benign", num_items_in_batch=num_items_in_batch)
            clear_memory()
            loss_harmless_p = backward_hf(loss_harmless_p)
            harmless_grad_p  = self.get_grad(model = model, device = self.grad_device2,zero_grad=True)

        # calculate the total gradient
        for name, param in (model.module.named_parameters() if hasattr(model, "module") else model.named_parameters()):
            if param.requires_grad:
                harmful_grad_p[name] = harmful_grad_p[name].to(harmful_grad[name].device)
                harmless_grad_p[name] = harmless_grad_p[name].to(harmless_grad[name].device)
                sld_weights = self.seam_args.beta / self.seam_args.epsilon
                if name not in self.total_grad.keys():
                    self.total_grad[name] = sld_weights * ((harmful_grad_p[name] - harmful_grad[name])/harmful_grad_norm + (harmless_grad_p[name] - harmless_grad[name])/harmless_grad_norm) 
                else:
                    self.total_grad[name] += sld_weights * ((harmful_grad_p[name] - harmful_grad[name])/harmful_grad_norm + (harmless_grad_p[name] - harmless_grad[name])/harmless_grad_norm) 
                del harmful_grad_p[name], harmless_grad_p[name], harmful_grad[name], harmless_grad[name]
        del harmful_grad_norm, harmless_grad_norm, harmful_grad_p, harmless_grad_p
        del harmless_grad, harmful_grad
        ####################################################################################
        del inputs
        clear_memory()

        return output_loss/self.args.gradient_accumulation_steps

    def compute_cosine_similarity(self, grad_dict_1, grad_dict_2):
        keys1 = set(grad_dict_1.keys())
        keys2 = set(grad_dict_2.keys())
        common_keys = list(keys1.intersection(keys2))
        dot_product = sum(torch.sum(grad_dict_1[name] * grad_dict_2[name]) for name in common_keys)
        norm_1_squared = sum(torch.sum(grad_dict_1[name] * grad_dict_1[name]) for name in common_keys)
        norm_2_squared = sum(torch.sum(grad_dict_2[name] * grad_dict_2[name]) for name in common_keys)
        similarity = dot_product / (torch.sqrt(norm_1_squared) * torch.sqrt(norm_2_squared) + 1e-8)
        return similarity
    
    
    def compute_l2_norm(self, grad_dict):
        return torch.sqrt(sum(torch.sum(g * g) for g in grad_dict.values()))

    def masked_token_ce_loss(self,
        logits,
        labels,
        mask
    ):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shifted masks
        shift_logit_mask = mask[..., :-1].contiguous()
        expanded_mask = shift_logit_mask.unsqueeze(-1).expand(-1, -1, shift_logits.size(-1))
        shift_label_mask = mask[..., 1:].contiguous()
        shift_logits = shift_logits * expanded_mask
        shift_labels = shift_labels * shift_label_mask
        shift_labels[shift_labels == 0] = -100
        shift_labels = shift_labels.type(torch.LongTensor).to(shift_logits.device)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
    def ce_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss
        
    def compute_loss(self, model, inputs, split = "harmful",return_outputs=False, num_items_in_batch=None):
        harmful_batch, harmless_batch, benign_batch = inputs[0], inputs[1], inputs[2]
        mask = ~torch.eq(harmful_batch['input_ids'], harmless_batch['input_ids'])
        mask = mask.float().to(harmful_batch['input_ids'].device)
        if split == "harmful": # calculate loss_ul, which is adadpted from https://github.com/domenicrosati/representation-noising/blob/main/representation_noising/loss.py
            harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
            harmful_outputs_loss = self.masked_token_ce_loss(
                harmful_outputs.logits,
                harmful_batch['labels'],
                mask
            )
            if hasattr(model, "module"):
                # For models wrapped in DataParallel
                output_embeddings = model.module.get_output_embeddings()
                norm = model.module.base_model.norm
            else:
                # For models not wrapped in DataParallel
                # This is for compatibility with models that have a custom output layer
                output_embeddings = model.get_output_embeddings()
                norm = model.base_model.norm

            harmful_losses = harmful_outputs_loss
            for i, h in enumerate(harmful_outputs.hidden_states):
                out = output_embeddings(norm(h).to(dtype=output_embeddings.weight.dtype))
                loss = self.masked_token_ce_loss(
                    out, harmful_batch['labels'],
                    mask
                )
                harmful_losses += loss
            harmful_losses = harmful_losses / len(harmful_outputs.hidden_states) + 1
            del mask, harmful_outputs, norm, output_embeddings, harmful_batch, harmless_batch, benign_batch
            loss =  - torch.log(harmful_losses)
        elif split == "harmless": # calculate loss_up
            harmless_outputs = model(harmless_batch['input_ids'], attention_mask=harmless_batch['attention_mask'],output_hidden_states=True)
            loss = self.masked_token_ce_loss(
                harmless_outputs.logits,
                harmless_batch['labels'],
                mask
            )
            del mask, harmless_outputs, harmful_batch, harmless_batch, benign_batch
        
        elif split == "attack": # calculate loss for harmful fine-tuning attack
            harmful_outputs = model(harmful_batch['input_ids'], attention_mask=harmful_batch['attention_mask'], output_hidden_states=True)
            loss = self.masked_token_ce_loss(
                harmful_outputs.logits,
                harmful_batch['labels'],
                mask
            )
            del mask, harmful_outputs, harmful_batch, harmless_batch, benign_batch
        elif split == "benign": # calculate sft loss for benign dataset
            benign_outputs = model(benign_batch['input_ids'], attention_mask=benign_batch['attention_mask'],output_hidden_states=True)
            loss = self.ce_loss(
                benign_outputs.logits,
                benign_batch['labels'],
            )
            del mask, benign_outputs, harmful_batch, harmless_batch, benign_batch
        
        assert self.accelerator.num_processes == 1, "UnlearnTrainer only supports single process training"
        return loss
    
