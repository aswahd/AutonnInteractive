"""Configuration for prompted-support automatic segmentation and joint supervision."""

from dataclasses import asdict, dataclass, field


@dataclass
class ModelConfig:
    patch_size: tuple[int, int, int] = (192, 192, 192)
    context_patch_size: tuple[int, int, int] | None = (192, 384, 384)
    full_volume: bool = False
    fixed_support_crop: bool = False
    support_mask_context: bool = True
    require_nonempty_support_masks: bool = True
    convert_dense_support_masks: bool = False
    support_mask_attention: bool = False
    query_deep_supervision: bool = True
    mask_encoder_stages: tuple[int, ...] | None = None
    support_position_encoding: bool = False
    query_position_encoding: bool = False
    rope_3d: bool = False
    interleave_query_context: bool = False
    cross_attention_type: str = "flat"
    support_aggregation: str = "learned"
    student_input_size: tuple[int, int, int] | None = None
    teacher_input_size: tuple[int, int, int] = (192, 192, 192)
    expected_channels: tuple[int, ...] = (32, 64, 128, 256, 320, 320)
    prompt_channels: int = 7
    stages: tuple[int, ...] = (-3, -2, -1)
    memory_grids: tuple[tuple[int, int, int] | None, ...] = (None, None, None)
    attention_dim: int = 256
    attention_heads: int = 4
    attention_depth: int = 4
    ffn_hidden: int = 512
    activation_checkpointing: bool = True
    support_chunk_size: int = 1
    support_encoder_chunk_size: int | None = None
    prototype_memory_grid: int | None = None

    def __post_init__(self):
        if self.student_input_size is not None:
            self.patch_size = tuple(self.student_input_size)
        for name in ("stages", "mask_encoder_stages"):
            stages = getattr(self, name)
            if stages is not None:
                setattr(self, name, tuple(s + len(self.expected_channels) if s < 0 else s for s in stages))

    @property
    def context_size(self):
        return tuple(self.patch_size if self.full_volume else self.context_patch_size or self.patch_size)

    @property
    def resized_context(self):
        return self.context_size != tuple(self.patch_size)

    def sampler_options(self):
        return {
            "patch_size": self.context_size,
            "student_input_size": self.patch_size,
            "full_volume": self.full_volume,
            "fixed_support_crop": self.fixed_support_crop,
        }

    def validate(self):
        if self.cross_attention_type not in ("flat", "hierarchical"):
            raise ValueError("cross_attention_type must be flat or hierarchical")
        if self.support_aggregation not in ("mean", "learned"):
            raise ValueError("support_aggregation must be mean or learned")
        if self.cross_attention_type == "hierarchical" and self.prototype_memory_grid is not None:
            raise ValueError("hierarchical attention requires direct support memory")
        if self.interleave_query_context and (
            self.prototype_memory_grid is not None or self.mask_encoder_stages is not None
        ):
            raise ValueError("interleaved query context requires direct memory and a trainable image encoder")
        if type(self.attention_depth) is not int or self.attention_depth < 1:
            raise ValueError("attention_depth must be a positive integer")
        if self.support_mask_attention and (not self.support_mask_context or self.prototype_memory_grid is not None):
            raise ValueError("masked attention requires converted masks and direct support memory")
        if self.prototype_memory_grid is not None and self.prototype_memory_grid < 1:
            raise ValueError("prototype memory grid must be positive")
        if self.resized_context and any(grid is not None for grid in self.memory_grids):
            raise ValueError("large-context students require unpooled support memory at all attention stages")
        if len(self.context_size) != 3 or any(n < 1 for n in self.context_size):
            raise ValueError("context size must contain three positive DHW dimensions")
        if len(self.teacher_input_size) != 3 or any(n < 32 or n % 32 for n in self.teacher_input_size):
            raise ValueError("teacher DHW dimensions must be positive multiples of 32")
        if len(self.expected_channels) != 6:
            raise ValueError("v2 requires a six-stage encoder")
        for stages in (self.stages, self.mask_encoder_stages):
            if stages is not None and (not stages or len(set(stages)) != len(stages) or not set(stages) <= {3, 4, 5}):
                raise ValueError("select distinct stages from the last three encoder stages")
        if self.mask_encoder_stages is not None and (
            not self.support_mask_context or not set(self.stages) <= set(self.mask_encoder_stages)
        ):
            raise ValueError("mask encoding requires frozen prompt conversion and heads at every attention stage")
        if self.prototype_memory_grid is not None and (
            self.mask_encoder_stages is not None or self.support_position_encoding or self.query_position_encoding
        ):
            raise ValueError("mask encoding and sinusoidal positions require direct support memory")
        if len(self.memory_grids) != len(self.stages) or self.attention_dim % self.attention_heads:
            raise ValueError("incompatible memory grids or attention dimensions")
        if self.rope_3d:
            if self.query_position_encoding or self.support_position_encoding or self.prototype_memory_grid is not None:
                raise ValueError("3D RoPE requires direct memory without additive position encodings")
            head_dim = self.attention_dim // self.attention_heads
            if head_dim < 6 or head_dim % 2:
                raise ValueError("3D RoPE requires an even head dimension of at least six")
        if any(size < 32 or size % 32 for size in self.patch_size):
            raise ValueError("patch axes must be positive multiples of 32")
        if self.prompt_channels != 7 or self.support_chunk_size < 1:
            raise ValueError("expected seven prompt channels and positive support chunk size")
        if self.support_encoder_chunk_size is not None and (
            type(self.support_encoder_chunk_size) is not int or self.support_encoder_chunk_size < 1
        ):
            raise ValueError("support encoder chunk size must be a positive integer")


@dataclass
class TrainConfig:
    seed: int = 2026
    updates: int = 50_000
    support_count: int = 4
    supervoxel_probability: float = 0.0
    support_dropout: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "min_active_supports": 1,
            "max_active_supports": 4,
            "sampling_strategy": "uniform_active_count",
            "per_episode": True,
            "count_probabilities": None,
        }
    )
    microbatch: int = 1
    effective_batch: int = 4
    backbone_lr: float = 1e-5
    context_lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_updates: int = 500
    final_lr_fraction: float = 0.1
    gradient_clip: float = 1.0
    patch_validation_every: int = 5000
    full_validation_every: int = -1
    validation_patches_per_target: int = 2
    validation_targets: list[str] | None = None
    checkpoint_every: int = 1000
    prompt_probabilities: dict[str, float] = field(
        default_factory=lambda: {
            "dense_mask": 0.4,
            "point": 0.15,
            "scribble": 0.15,
            "box": 0.15,
            "lasso": 0.15,
        }
    )
    crop_jitter: int = 48
    query_probabilities: tuple[float, float, float] = (0.5, 0.25, 0.25)
    sample_positive_queries: bool = False
    negative_attempts: int = 16
    augment_intensity: bool = True
    spatial_augmentation: dict | None = None
    spatial_sampling: dict | None = None
    prefetch_batches: bool = True
    num_workers: int = 2
    worker_backend: str = "thread"
    pin_memory: bool = True
    # Pending episodes per worker, independent of GPU microbatch size.
    prefetch_factor: int = 2

    lambda_support: float = 0.0
    supervise_supports: bool = False
    supervise_query: bool = True
    perturb_dense_supports: bool = True
    interactive_correction: dict[str, bool] = field(default_factory=lambda: {"enabled": False})

    def __post_init__(self):
        from .support_dropout import validate_dropout_config

        self.support_dropout = validate_dropout_config(self.support_dropout)
        if not 0 <= self.supervoxel_probability < 1:
            raise ValueError("supervoxel probability must lie in [0, 1)")
        if self.spatial_sampling is not None:
            from .spatial_sampling import SpatialSampling

            self.spatial_sampling = asdict(SpatialSampling(**self.spatial_sampling))
        if self.spatial_augmentation is not None:
            from .augmentation import SpatialAugmentation

            self.spatial_augmentation = asdict(SpatialAugmentation(**self.spatial_augmentation))
        if self.worker_backend not in {"thread", "process"}:
            raise ValueError("worker_backend must be thread or process")
        if self.updates < 1 or self.num_workers < 1 or self.prefetch_factor < 1:
            raise ValueError("updates, num_workers, and prefetch_factor must be positive")
        if self.patch_validation_every < 1 or self.validation_patches_per_target < 1 or self.checkpoint_every < 1:
            raise ValueError("patch validation interval, patches per target, and checkpoint interval must be positive")
        if self.validation_targets is not None and (
            not isinstance(self.validation_targets, list)
            or not self.validation_targets
            or any(not isinstance(t, str) for t in self.validation_targets)
            or len(set(self.validation_targets)) != len(self.validation_targets)
        ):
            raise ValueError("validation_targets must be a nonempty list of distinct target names")
        if self.full_validation_every != -1 and self.full_validation_every < 1:
            raise ValueError("full_validation_every must be positive or -1 to disable it")
        if self.lambda_support < 0 or not (self.supervise_supports or self.supervise_query):
            raise ValueError("at least one supervised pathway and a nonnegative support weight required")
        if self.interactive_correction != {"enabled": False}:
            raise NotImplementedError("Optional corrective interactions are disabled in the joint baseline")

    def accumulation_steps(self, world_size=1):
        divisor = self.microbatch * world_size
        if divisor < 1 or self.effective_batch < divisor or self.effective_batch % divisor:
            raise ValueError("effective batch must be divisible by microbatch * world_size")
        return self.effective_batch // divisor

    def to_dict(self):
        return asdict(self)

    @property
    def selection_metric(self):
        return "full_volume" if self.full_validation_every > 0 else "query_patch"

    def validation_kinds(self, completed, stopping=False):
        if stopping:
            return []
        final = completed == self.updates
        full = self.full_validation_every > 0 and (final or completed % self.full_validation_every == 0)
        patch = final or full or completed % self.patch_validation_every == 0
        return (["query_patch"] if patch else []) + (["full_volume"] if full else [])

    def resume_settings(self):
        """Runtime and validation settings may change without changing training episodes."""
        return {
            k: v
            for k, v in asdict(self).items()
            if k
            not in {
                "prefetch_batches",
                "num_workers",
                "prefetch_factor",
                "worker_backend",
                "pin_memory",
                "patch_validation_every",
                "full_validation_every",
                "validation_patches_per_target",
                "validation_targets",
                "checkpoint_every",
            }
        }
