from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from tr_asbs.variational import ASBSFourierControl, DISMethod, RolloutResult
from tr_asbs.variational.evaluation import EvaluationOutput
from tr_asbs.variational.trainer import VariationalTrainer
from tr_asbs.utils.helper import JointMetricDivergenceStopper


CONFIG_DIR = Path(__file__).parents[1] / "src" / "tr_asbs" / "config"


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "experiment",
    ["mw5", "mw5_delta", "gmm_dim", "gmm_mode"],
)
def test_variational_hydra_defaults_compose(experiment: str) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        cfg = compose(
            config_name="variational_train",
            overrides=[f"experiment={experiment}", "algorithm=pis"],
        )

    assert cfg.algorithm.iters == 40_000
    assert cfg.algorithm.batch_size == cfg.train_batch_size
    assert cfg.algorithm.num_steps == 128
    assert cfg.algorithm.optimizer.lr == 5e-4
    assert cfg.algorithm.grad_clip == 1.0
    assert cfg.algorithm.schedule.sigma_min == cfg.base_sde.sigma_min
    assert cfg.algorithm.schedule.s == 0.008
    assert cfg.algorithm.schedule.pow == 2
    assert cfg.algorithm.max_diffusion == cfg.base_sde.sigma_max
    assert cfg.algorithm.model._target_ == "tr_asbs.variational.ASBSFourierControl"
    assert cfg.algorithm.model.num_layers == 4
    assert cfg.algorithm.model.channels == 256
    assert cfg.algorithm.model.clip_norm == 100.0
    assert cfg.algorithm.model.score_hint_mode == "none"
    assert cfg.dtype == "float32"
    assert cfg.visualize is True
    assert cfg.evaluation.num_evaluations == 150
    assert cfg.checkpoint.interval == 5_000
    assert cfg.resume_from is None
    assert cfg.evaluation.evaluate_initial_state is True
    assert cfg.divergence_early_stop.min_evals == 20
    assert cfg.divergence_early_stop.primary_factor == 2.0
    assert cfg.divergence_early_stop.secondary_delta == 0.15
    assert cfg.divergence_early_stop.patience == 8
    assert cfg.divergence_early_stop.nonfinite_patience == 2
    assert cfg.run_name.endswith("_pis")
    assert "wandb" in cfg.fabric.loggers._target_.lower()
    assert cfg.fabric.loggers.project == "tr_asbs"


@pytest.mark.parametrize(
    ("experiment", "expected_init_std"),
    [("mw5", 1.0), ("gmm_mode", 30.0)],
)
def test_dis_hydra_defaults_compose(
    experiment: str,
    expected_init_std: float,
) -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        cfg = compose(
            config_name="variational_train",
            overrides=[f"experiment={experiment}", "algorithm=dis"],
        )

    assert cfg.algorithm.name == "dis"
    assert cfg.algorithm.init_std == expected_init_std
    assert cfg.algorithm.init_std == cfg.source.scale
    assert cfg.algorithm.method._target_ == "tr_asbs.variational.DISMethod"
    assert cfg.algorithm.model._target_ == "tr_asbs.variational.ASBSFourierControl"
    assert cfg.algorithm.model.score_hint_mode == "none"
    assert cfg.algorithm.max_diffusion == cfg.base_sde.sigma_max
    assert cfg.run_name.endswith("_dis")


def test_gmm_target_is_fixed_while_model_seed_changes_initialization() -> None:
    targets: list[tuple[torch.Tensor, torch.Tensor]] = []
    parameters: list[torch.Tensor] = []
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        for seed in (0, 1):
            cfg = compose(
                config_name="variational_train",
                overrides=[
                    "experiment=gmm_mode",
                    "algorithm=pis",
                    f"seed={seed}",
                    "+potential.seed=0",
                ],
            )
            torch.manual_seed(seed)
            target = hydra.utils.instantiate(cfg.potential, device="cpu")
            control = hydra.utils.instantiate(
                cfg.algorithm.model,
                dim=target.dim,
            )
            targets.append((target.locs.clone(), target.cat_probs.clone()))
            parameters.append(
                torch.cat(
                    [
                        parameter.detach().reshape(-1)
                        for parameter in control.parameters()
                    ]
                )
            )

    torch.testing.assert_close(targets[0][0], targets[1][0])
    torch.testing.assert_close(targets[0][1], targets[1][1])
    assert not torch.equal(parameters[0], parameters[1])


def test_nonfinite_diagnostics_and_gradient_sanitizing() -> None:
    trainer = object.__new__(VariationalTrainer)
    trainer.iteration = 13
    result = RolloutResult(
        samples=torch.tensor([[float("nan")]]),
        running_cost=torch.zeros(1),
        stochastic_cost=torch.zeros(1),
        terminal_cost=torch.zeros(1),
    )

    with pytest.raises(
        FloatingPointError,
        match=r"iteration 13: samples=1",
    ):
        trainer._validate_rollout(result, "test")

    trainer.method = torch.nn.Linear(3, 1, bias=False)
    trainer.method.weight.grad = torch.tensor([[float("nan"), float("inf"), 2.0]])
    count = trainer._sanitize_gradients()

    assert count == 2
    torch.testing.assert_close(
        trainer.method.weight.grad,
        torch.tensor([[0.0, 0.0, 2.0]]),
    )


def test_checkpoint_monitors_support_mixed_min_and_max_modes() -> None:
    trainer = object.__new__(VariationalTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "monitor_values": ["loss/SD", "eval/raw_path_ess"],
            "monitor_modes": {
                "loss/SD": "min",
                "eval/raw_path_ess": "max",
            },
            "checkpoint": {"mode": "min"},
        }
    )
    trainer.best_metrics = {}
    saved: list[str] = []
    trainer.save_checkpoint = lambda filename: saved.append(filename)

    trainer._update_best_checkpoints({"loss/SD": 2.0, "eval/raw_path_ess": 0.2})
    trainer._update_best_checkpoints({"loss/SD": 3.0, "eval/raw_path_ess": 0.4})

    assert trainer.best_metrics == {
        "loss/SD": 2.0,
        "eval/raw_path_ess": 0.4,
    }
    assert saved == [
        "loss_SD_best.pt",
        "eval_raw_path_ess_best.pt",
        "eval_raw_path_ess_best.pt",
    ]


def test_variational_evaluation_uses_shared_divergence_guard() -> None:
    trainer = object.__new__(VariationalTrainer)
    trainer.iteration = 200
    trainer.evaluation_count = 20
    trainer.early_stop_requested = False
    trainer._wandb_logging = False
    trainer.logger = None
    trainer.divergence_stopper = JointMetricDivergenceStopper(
        {
            "enabled": True,
            "primary_metric": "loss/SD",
            "secondary_metric": "loss/tvd_soft",
            "min_evals": 20,
            "primary_factor": 2.0,
            "secondary_delta": 0.15,
            "patience": 1,
            "nonfinite_patience": 2,
        }
    )
    trainer.divergence_stopper.best_primary = 1.0
    trainer.divergence_stopper.best_secondary = 0.1
    trainer._log = lambda metrics: None
    trainer._log_media = lambda media: None
    trainer._update_best_checkpoints = lambda metrics: None
    output = EvaluationOutput(
        metrics={"loss/SD": 2.1, "loss/tvd_soft": 0.26},
        samples=torch.zeros(1, 1),
        log_weights=torch.zeros(1),
    )

    trainer._process_evaluation(output, update_checkpoints=True)

    assert trainer.early_stop_requested is True
    assert trainer.evaluation_count == 21
    assert trainer.divergence_stopper.reason is not None


def _compose_small_run(
    *,
    experiment: str,
    output_directory: Path,
    iterations: int,
    resume_from: Path | None = None,
    algorithm: str = "pis",
):
    overrides = [
        f"experiment={experiment}",
        f"algorithm={algorithm}",
        "fabric.accelerator=cpu",
        "fabric.devices=1",
        "fabric.strategy=auto",
        "fabric.loggers._target_=lightning.fabric.loggers.CSVLogger",
        f"fabric.loggers.save_dir={output_directory / 'logs'}",
        f"+fabric.loggers.root_dir={output_directory / 'logs'}",
        "progress_bar=false",
        "log_interval=1",
        "monitor_values=[]",
        f"algorithm.iters={iterations}",
        "algorithm.num_steps=2",
        "algorithm.batch_size=4",
        "algorithm.microbatch_size=2",
        "algorithm.model.num_layers=2",
        "algorithm.model.channels=8",
        "evaluation.num_evaluations=1",
        "evaluation.num_samples=8",
        "evaluation.final_num_samples=8",
        "evaluation.chunk_size=4",
        "evaluation.include_target_details=false",
        f"checkpoint.directory={output_directory / 'checkpoints'}",
        "checkpoint.interval=1",
    ]
    if resume_from is not None:
        overrides.append(f"resume_from={resume_from}")

    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR.resolve()),
    ):
        return compose(
            config_name="variational_train",
            overrides=overrides,
        )


@pytest.mark.parametrize(
    ("experiment", "expected_max_diffusion"),
    [("mw5", 0.2), ("gmm_mode", 2.0)],
)
def test_hydra_training_checkpoint_and_resume_smoke(
    tmp_path: Path,
    experiment: str,
    expected_max_diffusion: float,
) -> None:
    output_directory = tmp_path / experiment
    first_cfg = _compose_small_run(
        experiment=experiment,
        output_directory=output_directory,
        iterations=1,
    )

    assert first_cfg.algorithm.iters == 1
    assert first_cfg.algorithm.batch_size == 4
    assert first_cfg.run_name.endswith("_pis")
    assert first_cfg.evaluation.num_evaluations == 1

    first_trainer = VariationalTrainer(first_cfg)
    assert isinstance(first_trainer.method.control, ASBSFourierControl)
    assert first_trainer.method.control.uses_score_hint is False
    assert first_trainer.method.noise_schedule.sigma_max == (expected_max_diffusion)
    first_metrics = first_trainer.fit()

    checkpoint = output_directory / "checkpoints" / "last.pt"
    assert checkpoint.is_file()
    assert (output_directory / "checkpoints" / "step_00000001.pt").is_file()
    assert (output_directory / "checkpoints" / "eval_neg_elbo_best.pt").is_file()
    assert "eval/neg_elbo" in first_metrics

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["iteration"] == 1
    assert set(saved) == {
        "model",
        "optimizer",
        "iteration",
        "best_metrics",
        "evaluation_count",
        "divergence_state",
        "cpu_rng_state",
        "cuda_rng_state_all",
    }

    torch.set_rng_state(saved["cpu_rng_state"])
    expected_random_values = torch.rand(6)

    resumed_cfg = _compose_small_run(
        experiment=experiment,
        output_directory=output_directory,
        iterations=2,
        resume_from=checkpoint,
    )
    resumed_trainer = VariationalTrainer(resumed_cfg)
    resumed_trainer.load_checkpoint(checkpoint)
    actual_random_values = torch.rand(6)
    torch.testing.assert_close(actual_random_values, expected_random_values)

    # Reset to the checkpoint RNG once more before fitting, since the equality
    # check above intentionally consumed random values.
    resumed_trainer.load_checkpoint(checkpoint)
    resumed_metrics = resumed_trainer.fit()

    final_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert final_state["iteration"] == 2
    assert (output_directory / "checkpoints" / "step_00000002.pt").is_file()
    assert "eval/log_z" in resumed_metrics
    assert list((output_directory / "logs").glob("**/metrics.csv"))


def test_dis_hydra_training_smoke(tmp_path: Path) -> None:
    output_directory = tmp_path / "dis"
    cfg = _compose_small_run(
        experiment="mw5",
        output_directory=output_directory,
        iterations=1,
        algorithm="dis",
    )

    trainer = VariationalTrainer(cfg)
    assert isinstance(trainer.method._forward_module, DISMethod)
    assert trainer.method.init_std == cfg.source.scale

    metrics = trainer.fit()

    assert "eval/neg_elbo" in metrics
    assert (output_directory / "checkpoints" / "last.pt").is_file()
    assert (output_directory / "checkpoints" / "checkpoint_evaluations.json").is_file()
