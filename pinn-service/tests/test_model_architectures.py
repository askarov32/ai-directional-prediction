from __future__ import annotations

import torch

from pinn_service.losses import compute_hybrid_pinn_loss
from pinn_service.model import MLP_PINN, ResSplitPINN, create_pinn_model, parse_layer_dims
from pinn_service.train import build_parser as build_train_parser


def test_baseline_mlp_still_returns_expected_shape():
    model = MLP_PINN(input_dim=10, output_dim=4, hidden_dim=192, depth=6, activation="tanh")
    inputs = torch.randn(5, 10)
    outputs = model(inputs)

    assert outputs.shape == (5, 4)


def test_res_split_accepts_public_input_contract_and_returns_public_output_contract():
    model = ResSplitPINN(
        input_dim=10,
        output_dim=4,
        hidden_dim=192,
        num_blocks=4,
        activation="tanh",
    )
    inputs = torch.randn(7, 10)
    outputs = model(inputs)

    assert outputs.shape == (7, 4)
    assert model.temperature_head[-1].out_features == 1
    assert model.displacement_head[-1].out_features == 3


def test_fourier_features_keep_the_same_public_input_shape():
    model = create_pinn_model(
        input_dim=10,
        output_dim=4,
        architecture="res_split",
        hidden_dim=192,
        num_blocks=4,
        activation="tanh",
        use_fourier_features=True,
        fourier_num_frequencies=4,
        fourier_scale=1.0,
    )
    inputs = torch.randn(3, 10)
    outputs = model(inputs)

    assert outputs.shape == (3, 4)


def test_train_parser_accepts_res_split_configuration():
    parser = build_train_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "train.npz",
            "--output-dir",
            "out",
            "--architecture",
            "res_split",
            "--hidden-dim",
            "256",
            "--num-blocks",
            "5",
            "--activation",
            "silu",
            "--use-fourier-features",
            "--fourier-num-frequencies",
            "6",
            "--fourier-scale",
            "1.5",
        ]
    )

    assert args.architecture == "res_split"
    assert args.num_blocks == 5
    assert args.use_fourier_features is True


def test_parse_layer_dims_supports_tapered_baseline_variants():
    assert parse_layer_dims("256,256,192,192,128,128") == (256, 256, 192, 192, 128, 128)


def test_hybrid_loss_runs_with_both_architectures():
    input_scaler_mean = torch.tensor(
        [0.5, 0.5, 0.0, 0.1, 4.0e10, 0.25, 2650.0, 7.5e-6, 2.7, 790.0],
        dtype=torch.float32,
    )
    input_scaler_std = torch.tensor([0.2, 0.2, 0.1, 0.05, 1.0e9, 0.05, 20.0, 1.0e-6, 0.3, 15.0], dtype=torch.float32)
    output_scaler_mean = torch.tensor([293.15, 0.0, 0.0, 0.0], dtype=torch.float32)
    output_scaler_std = torch.tensor([10.0, 1.0e-3, 1.0e-3, 1.0e-3], dtype=torch.float32)

    inputs_scaled = torch.tensor(
        [
            [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, -0.4, 0.2, -0.2, 0.1, -0.2, 0.2, 0.1, -0.3, 0.1],
            [-0.6, 0.3, -0.1, 0.8, -0.15, 0.15, -0.1, 0.0, 0.2, -0.2],
            [0.2, 0.7, 0.1, 0.4, 0.05, 0.05, -0.05, -0.1, 0.1, 0.0],
        ],
        dtype=torch.float32,
    )
    primary_targets_scaled = torch.zeros(4, 4, dtype=torch.float32)
    velocity_targets = torch.zeros(4, 3, dtype=torch.float32)

    for architecture in ("mlp", "res_split"):
        model = create_pinn_model(
            input_dim=10,
            output_dim=4,
            architecture=architecture,
            hidden_dim=96,
            depth=3,
            num_blocks=2,
            activation="tanh",
        )
        loss, metrics = compute_hybrid_pinn_loss(
            model=model,
            inputs_scaled=inputs_scaled.clone(),
            primary_targets_scaled=primary_targets_scaled,
            velocity_targets=velocity_targets,
            input_scaler_mean=input_scaler_mean,
            input_scaler_std=input_scaler_std,
            output_scaler_mean=output_scaler_mean,
            output_scaler_std=output_scaler_std,
            supervised_weight=1.0,
            velocity_weight=0.25,
            wave_residual_weight=0.1,
            thermal_residual_weight=0.05,
            reference_temperature_k=293.15,
            physics_mode="coupled_thermoelastic",
            loss_balance_mode="fixed",
        )

        assert torch.isfinite(loss)
        assert metrics["total_loss"] >= 0.0
