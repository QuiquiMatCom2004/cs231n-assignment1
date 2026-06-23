import numpy as np
import pytest
from cs231n.solver import Solver
from cs231n.classifiers.fc_net import TwoLayerNet


def make_toy_data(N=50, D=10, C=3):
    np.random.seed(0)
    X = np.random.randn(N, D)
    y = np.random.randint(0, C, N)
    return {
        "X_train": X[:40],
        "y_train": y[:40],
        "X_val": X[40:],
        "y_val": y[40:],
    }


class TestSolverInit:
    def test_basic_init(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(model, data, verbose=False, num_epochs=1)
        assert solver.num_epochs == 1

    def test_invalid_update_rule_raises(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        with pytest.raises(ValueError):
            Solver(model, data, update_rule="nonexistent")

    def test_extra_kwargs_raises(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        with pytest.raises(ValueError):
            Solver(model, data, bogus_param=42)


class TestSolverTrain:
    def test_loss_decreases(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(
            model,
            data,
            verbose=False,
            num_epochs=20,
            batch_size=20,
            optim_config={"learning_rate": 1e-2},
        )
        solver.train()
        assert len(solver.loss_history) > 0
        first_losses = np.mean(solver.loss_history[:5])
        last_losses = np.mean(solver.loss_history[-5:])
        assert last_losses < first_losses

    def test_accuracy_histories_populated(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(
            model,
            data,
            verbose=False,
            num_epochs=3,
            batch_size=20,
            optim_config={"learning_rate": 1e-2},
        )
        solver.train()
        assert len(solver.train_acc_history) > 0
        assert len(solver.val_acc_history) > 0

    def test_best_params_set(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(
            model,
            data,
            verbose=False,
            num_epochs=5,
            batch_size=20,
            optim_config={"learning_rate": 1e-2},
        )
        solver.train()
        assert "W1" in model.params
        assert "W2" in model.params

    def test_lr_decay(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(
            model,
            data,
            verbose=False,
            num_epochs=3,
            batch_size=40,
            lr_decay=0.5,
            optim_config={"learning_rate": 1e-2},
        )
        solver.train()
        final_lr = solver.optim_configs["W1"]["learning_rate"]
        assert final_lr < 1e-2


class TestSolverCheckAccuracy:
    def test_accuracy_in_range(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(model, data, verbose=False, num_epochs=1)
        acc = solver.check_accuracy(data["X_val"], data["y_val"])
        assert 0 <= acc <= 1

    def test_subsampling(self):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(model, data, verbose=False, num_epochs=1)
        acc = solver.check_accuracy(data["X_train"], data["y_train"], num_samples=5)
        assert 0 <= acc <= 1


class TestSolverUpdateRules:
    @pytest.mark.parametrize("rule", ["sgd", "sgd_momentum", "rmsprop", "adam"])
    def test_different_update_rules(self, rule):
        data = make_toy_data()
        model = TwoLayerNet(input_dim=10, hidden_dim=8, num_classes=3)
        solver = Solver(
            model,
            data,
            verbose=False,
            num_epochs=2,
            batch_size=20,
            update_rule=rule,
            optim_config={"learning_rate": 1e-3},
        )
        solver.train()
        assert len(solver.loss_history) > 0
