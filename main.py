"""
TensorFlow-Quantum (TFQ) MNIST Classification
==============================================
A comprehensive quantum machine learning project that implements 5 different
quantum circuit architectures for classifying MNIST digits using TensorFlow-Quantum.

Architectures:
    1. BasicQuantumCircuit      - Simple single-layer rotation gates
    2. EntangledQuantumCircuit  - CNOT-based entanglement layer
    3. LayeredVQC               - Multi-layer variational quantum circuit
    4. HybridDeepQNN            - Deep hybrid quantum-classical network
    5. AnsatzQuantumCircuit     - Hardware-efficient ansatz circuit

Metrics: Balanced Accuracy, Precision, Recall, F1 (macro)

Requirements:
    pip install tensorflow==2.15.0 tensorflow-quantum cirq sympy
    pip install scikit-learn matplotlib seaborn numpy

Usage:
    python main.py
"""

# ─────────────────────────────── Standard Library ────────────────────────────
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

# Suppress TF/Cirq verbosity before importing
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ─────────────────────────────── Third-Party ─────────────────────────────────
try:
    import cirq
    import matplotlib
    import numpy as np
    import sympy
    import tensorflow as tf
    import tensorflow_quantum as tfq

    matplotlib.use("Agg")  # non-interactive backend — safe on all envs
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.preprocessing import LabelBinarizer
except ImportError as exc:
    sys.exit(
        f"[FATAL] Missing dependency: {exc}\n"
        "Install with: pip install tensorflow tensorflow-quantum cirq sympy "
        "scikit-learn matplotlib seaborn"
    )

# ─────────────────────────────── Global Config ───────────────────────────────
SEED: int = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Dataset settings
N_CLASSES: int = 10  # MNIST has 10 digit classes (0-9)
N_TRAIN: int = 1000  # Subset size for tractable quantum simulation
N_TEST: int = 200
IMG_SIZE: int = 28
N_PCA_COMPONENTS: int = 10  # PCA dims before binary encoding → qubits

# Training settings
EPOCHS: int = 10
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.01

# Output
OUTPUT_DIR: Path = Path("tfq_mnist_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colour palette used across all plots
PALETTE = sns.color_palette("husl", N_CLASSES)

# ─────────────────────────────── Logging ─────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 – DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════


def load_and_preprocess_mnist(
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    n_pca: int = N_PCA_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST, apply PCA dimensionality reduction, and binarise pixel values.

    The quantum encoding pipeline:
        raw (28×28) → flatten (784) → PCA (n_pca) → threshold binarise → qubits

    Args:
        n_train: Number of training samples to keep.
        n_test:  Number of test samples to keep.
        n_pca:   Number of PCA components (= number of qubits per sample).

    Returns:
        x_train, y_train, x_test, y_test as numpy arrays.
        x arrays have shape (N, n_pca) with float32 values in [0, 1].
        y arrays have shape (N,)       with int labels in {0, …, 9}.

    Raises:
        ValueError: If requested sample counts exceed dataset size.
        RuntimeError: If PCA fitting fails.
    """
    log.info("Loading MNIST dataset …")
    (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = tf.keras.datasets.mnist.load_data()

    # ── Validation ──────────────────────────────────────────────────────────
    if n_train > len(x_tr_raw):
        raise ValueError(
            f"n_train={n_train} exceeds available training samples ({len(x_tr_raw)})."
        )
    if n_test > len(x_te_raw):
        raise ValueError(
            f"n_test={n_test} exceeds available test samples ({len(x_te_raw)})."
        )
    if not (1 <= n_pca <= 784):
        raise ValueError(f"n_pca must be in [1, 784], got {n_pca}.")

    # ── Subset & normalise ───────────────────────────────────────────────────
    # Stratified sampling to keep class balance in the tiny subset
    train_idx = _stratified_indices(y_tr_raw, n_train)
    test_idx = _stratified_indices(y_te_raw, n_test)

    x_tr = x_tr_raw[train_idx].reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = y_tr_raw[train_idx].astype(np.int32)
    x_te = x_te_raw[test_idx].reshape(-1, 784).astype(np.float32) / 255.0
    y_te = y_te_raw[test_idx].astype(np.int32)

    # ── PCA ─────────────────────────────────────────────────────────────────
    log.info(f"Applying PCA: 784 → {n_pca} components …")
    try:
        pca = PCA(n_components=n_pca, random_state=SEED)
        x_tr_pca = pca.fit_transform(x_tr).astype(np.float32)
        x_te_pca = pca.transform(x_te).astype(np.float32)
    except Exception as exc:
        raise RuntimeError(f"PCA fitting failed: {exc}") from exc

    # ── Normalise PCA output to [0, π] for angle encoding ───────────────────
    # Each feature is independently scaled to [0, π] so it can be used as a
    # rotation angle in quantum gates (RX, RY, RZ gates accept radians).
    for dim in range(n_pca):
        col_min = x_tr_pca[:, dim].min()
        col_max = x_tr_pca[:, dim].max()
        span = col_max - col_min + 1e-8  # avoid division-by-zero
        x_tr_pca[:, dim] = (x_tr_pca[:, dim] - col_min) / span * np.pi
        x_te_pca[:, dim] = (x_te_pca[:, dim] - col_min) / span * np.pi

    log.info(
        f"Data ready │ train={x_tr_pca.shape}, test={x_te_pca.shape} │ "
        f"classes={np.unique(y_tr)}"
    )
    return x_tr_pca, y_tr, x_te_pca, y_te


def _stratified_indices(labels: np.ndarray, n: int) -> np.ndarray:
    """
    Return n indices drawn uniformly across all label classes.

    Args:
        labels: 1-D integer label array.
        n:      Total number of indices to return.

    Returns:
        1-D index array of length n.
    """
    classes = np.unique(labels)
    per_class = max(1, n // len(classes))
    idx_list = []
    rng = np.random.default_rng(SEED)
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        chosen = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        idx_list.append(chosen)
    all_idx = np.concatenate(idx_list)
    # Trim or pad to exactly n
    all_idx = rng.choice(all_idx, size=min(n, len(all_idx)), replace=False)
    return all_idx


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 – QUANTUM CIRCUIT BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each builder follows the same signature:
#      (qubits, symbols) → cirq.Circuit
#
#  The circuits are parameterised; `symbols` is a flat list of sympy.Symbol
#  objects that will later be resolved (trained) by TFQ optimizers.


def _make_symbols(prefix: str, count: int) -> list[sympy.Symbol]:
    """
    Create a list of unique sympy symbols with a given prefix.

    Args:
        prefix: Name prefix, e.g. "theta".
        count:  Number of symbols to create.

    Returns:
        List of sympy.Symbol objects.
    """
    return [sympy.Symbol(f"{prefix}_{i}") for i in range(count)]


# ── Architecture 1: Basic Quantum Circuit ─────────────────────────────────────


def build_basic_circuit(
    qubits: list[cirq.GridQubit],
    symbols: list[sympy.Symbol],
) -> cirq.Circuit:
    """
    Architecture 1 – BasicQuantumCircuit.

    A minimal single-layer circuit:
        • RX rotation on each qubit (data encoding)
        • RY rotation on each qubit (trainable parameter)
        • Z measurement on all qubits

    This architecture has 1 trainable parameter per qubit.

    Args:
        qubits:  List of cirq.GridQubit objects.
        symbols: Trainable sympy symbols (length = n_qubits).

    Returns:
        A cirq.Circuit representing the basic quantum model.
    """
    n = len(qubits)
    if len(symbols) < n:
        raise ValueError(f"Need ≥{n} symbols for BasicCircuit, got {len(symbols)}.")

    circuit = cirq.Circuit()

    # Layer 1: Data-encoding rotations (Hadamard + RZ encoding)
    circuit += [cirq.H(q) for q in qubits]

    # Layer 2: Trainable single-qubit RY rotations
    circuit += [cirq.ry(symbols[i])(qubits[i]) for i in range(n)]

    return circuit


def _n_params_basic(n_qubits: int) -> int:
    return n_qubits


# ── Architecture 2: Entangled Quantum Circuit ──────────────────────────────────


def build_entangled_circuit(
    qubits: list[cirq.GridQubit],
    symbols: list[sympy.Symbol],
) -> cirq.Circuit:
    """
    Architecture 2 – EntangledQuantumCircuit.

    Adds CNOT entanglement between adjacent qubits after the rotation layer:
        • H on each qubit
        • RY with trainable parameter
        • CNOT ladder: q0→q1, q1→q2, …, q(n-2)→q(n-1)
        • Second RY with trainable parameter

    This architecture has 2 trainable parameters per qubit.

    Args:
        qubits:  List of cirq.GridQubit objects.
        symbols: Trainable sympy symbols (length = 2 * n_qubits).

    Returns:
        A cirq.Circuit with entanglement structure.
    """
    n = len(qubits)
    required = 2 * n
    if len(symbols) < required:
        raise ValueError(
            f"EntangledCircuit requires ≥{required} symbols, got {len(symbols)}."
        )

    circuit = cirq.Circuit()

    # Layer 1: Superposition
    circuit += [cirq.H(q) for q in qubits]

    # Layer 2: First trainable rotation
    circuit += [cirq.ry(symbols[i])(qubits[i]) for i in range(n)]

    # Layer 3: Entanglement (nearest-neighbour CNOT ladder)
    circuit += [cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(n - 1)]

    # Layer 4: Second trainable rotation after entanglement
    circuit += [cirq.ry(symbols[n + i])(qubits[i]) for i in range(n)]

    return circuit


def _n_params_entangled(n_qubits: int) -> int:
    return 2 * n_qubits


# ── Architecture 3: Layered VQC ────────────────────────────────────────────────


def build_layered_vqc(
    qubits: list[cirq.GridQubit],
    symbols: list[sympy.Symbol],
    n_layers: int = 3,
) -> cirq.Circuit:
    """
    Architecture 3 – LayeredVQC (Variational Quantum Circuit).

    Repeats `n_layers` identical blocks, each consisting of:
        • RX rotation (trainable)
        • RZ rotation (trainable)
        • Ring-topology CNOT entanglement (last qubit connects back to first)

    This architecture has 2 * n_qubits * n_layers trainable parameters.

    Args:
        qubits:   List of cirq.GridQubit objects.
        symbols:  Trainable sympy symbols.
        n_layers: Number of variational blocks (depth).

    Returns:
        A deep cirq.Circuit with layered structure.
    """
    n = len(qubits)
    required = 2 * n * n_layers
    if len(symbols) < required:
        raise ValueError(
            f"LayeredVQC requires ≥{required} symbols for {n_layers} layers, "
            f"got {len(symbols)}."
        )

    circuit = cirq.Circuit()
    sym_idx = 0

    for layer in range(n_layers):
        # RX layer
        circuit += [cirq.rx(symbols[sym_idx + i])(qubits[i]) for i in range(n)]
        sym_idx += n

        # RZ layer
        circuit += [cirq.rz(symbols[sym_idx + i])(qubits[i]) for i in range(n)]
        sym_idx += n

        # Ring CNOT: q0→q1, q1→q2, …, q(n-1)→q0
        circuit += [cirq.CNOT(qubits[i], qubits[(i + 1) % n]) for i in range(n)]

    return circuit


def _n_params_layered_vqc(n_qubits: int, n_layers: int = 3) -> int:
    return 2 * n_qubits * n_layers


# ── Architecture 4: Hybrid Deep QNN ───────────────────────────────────────────


def build_hybrid_deep_qnn(
    qubits: list[cirq.GridQubit],
    symbols: list[sympy.Symbol],
) -> cirq.Circuit:
    """
    Architecture 4 – HybridDeepQNN.

    A richer ansatz that combines three rotation axes (RX, RY, RZ) with
    full-range two-qubit interactions (CZ gates), followed by a second
    single-qubit rotation pass:

        • RX + RY + RZ per qubit  (3 params/qubit)
        • CZ between all adjacent pairs
        • RY + RZ per qubit        (2 params/qubit)

    Total: 5 * n_qubits trainable parameters.

    Args:
        qubits:  List of cirq.GridQubit objects.
        symbols: Trainable sympy symbols (length = 5 * n_qubits).

    Returns:
        A hybrid deep quantum circuit.
    """
    n = len(qubits)
    required = 5 * n
    if len(symbols) < required:
        raise ValueError(
            f"HybridDeepQNN requires ≥{required} symbols, got {len(symbols)}."
        )

    circuit = cirq.Circuit()
    idx = 0

    # Block 1: All-axis single-qubit rotations
    circuit += [cirq.rx(symbols[idx + i])(qubits[i]) for i in range(n)]
    idx += n
    circuit += [cirq.ry(symbols[idx + i])(qubits[i]) for i in range(n)]
    idx += n
    circuit += [cirq.rz(symbols[idx + i])(qubits[i]) for i in range(n)]
    idx += n

    # Block 2: CZ entanglement (phase-kickback style)
    circuit += [cirq.CZ(qubits[i], qubits[i + 1]) for i in range(n - 1)]

    # Block 3: Post-entanglement refinement
    circuit += [cirq.ry(symbols[idx + i])(qubits[i]) for i in range(n)]
    idx += n
    circuit += [cirq.rz(symbols[idx + i])(qubits[i]) for i in range(n)]
    idx += n

    return circuit


def _n_params_hybrid_deep(n_qubits: int) -> int:
    return 5 * n_qubits


# ── Architecture 5: Hardware-Efficient Ansatz ─────────────────────────────────


def build_ansatz_circuit(
    qubits: list[cirq.GridQubit],
    symbols: list[sympy.Symbol],
    n_layers: int = 2,
) -> cirq.Circuit:
    """
    Architecture 5 – AnsatzQuantumCircuit (Hardware-Efficient Ansatz).

    Implements a hardware-efficient ansatz commonly used on real quantum
    processors. The structure alternates between:
        • Ry+Rz rotation block (2 params/qubit)
        • Odd-even CNOT pattern (qubits 0,1 │ qubits 2,3 │ …)
        • Even-odd CNOT pattern (qubits 1,2 │ qubits 3,4 │ …)

    This staggered entanglement avoids qubit connectivity constraints
    that arise on real hardware.

    Args:
        qubits:   List of cirq.GridQubit objects.
        symbols:  Trainable sympy symbols.
        n_layers: Number of ansatz repetitions.

    Returns:
        A hardware-efficient ansatz circuit.
    """
    n = len(qubits)
    required = 2 * n * n_layers
    if len(symbols) < required:
        raise ValueError(
            f"AnsatzCircuit requires ≥{required} symbols, got {len(symbols)}."
        )

    circuit = cirq.Circuit()
    idx = 0

    # Initial Hadamard wall to create superposition
    circuit += [cirq.H(q) for q in qubits]

    for _ in range(n_layers):
        # Rotation block
        circuit += [cirq.ry(symbols[idx + i])(qubits[i]) for i in range(n)]
        idx += n
        circuit += [cirq.rz(symbols[idx + i])(qubits[i]) for i in range(n)]
        idx += n

        # Odd-indexed CNOT pairs:  (0,1), (2,3), (4,5) …
        circuit += [cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(0, n - 1, 2)]
        # Even-indexed CNOT pairs: (1,2), (3,4), (5,6) …
        circuit += [cirq.CNOT(qubits[i], qubits[i + 1]) for i in range(1, n - 1, 2)]

    return circuit


def _n_params_ansatz(n_qubits: int, n_layers: int = 2) -> int:
    return 2 * n_qubits * n_layers


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 – TFQ MODEL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════


def encode_data_as_circuits(
    data: np.ndarray,
    qubits: list[cirq.GridQubit],
) -> tf.Tensor:
    """
    Encode classical feature vectors into quantum state-preparation circuits.

    Each feature value (already in [0, π]) is used as the angle of an RX gate
    applied to the corresponding qubit.  This is the "angle encoding" strategy:
        |ψ_i⟩ = RX(x_i) |0⟩

    Args:
        data:   Float array of shape (N, n_features).  Values in [0, π].
        qubits: List of n_features qubits.

    Returns:
        TFQ tensor of serialised cirq circuits, shape (N,).

    Raises:
        ValueError: If data.shape[1] does not match len(qubits).
    """
    n_features = data.shape[1]
    if n_features != len(qubits):
        raise ValueError(
            f"Data has {n_features} features but only {len(qubits)} qubits "
            "were provided."
        )

    circuits = []
    for sample in data:
        # Build per-sample encoding circuit
        encoding_circuit = cirq.Circuit(
            cirq.rx(float(sample[i]))(qubits[i]) for i in range(n_features)
        )
        circuits.append(encoding_circuit)

    return tfq.convert_to_tensor(circuits)


def build_tfq_model(
    circuit_builder,
    n_qubits: int,
    n_params: int,
    n_classes: int = N_CLASSES,
    learning_rate: float = LEARNING_RATE,
) -> tuple[tf.keras.Model, list[sympy.Symbol]]:
    """
    Construct a hybrid quantum-classical Keras model with TFQ.

    Architecture:
        Input (serialised circuits)
        → tfq.layers.PQC  (quantum layer; outputs expectation values)
        → Dense(n_classes, softmax)  (classical readout)

    The PQC layer runs the parameterised quantum circuit and returns
    ⟨Z⟩ expectation values on each qubit as classical real numbers.

    Args:
        circuit_builder: A callable (qubits, symbols) → cirq.Circuit.
        n_qubits:        Number of qubits (= number of PCA components).
        n_params:        Number of trainable parameters in the circuit.
        n_classes:       Output classes (10 for MNIST).
        learning_rate:   Adam optimiser learning rate.

    Returns:
        (model, symbols) where model is a compiled tf.keras.Model and
        symbols is the list of sympy.Symbol objects used in the circuit.

    Raises:
        ValueError: If circuit_builder produces unexpected output.
        RuntimeError: If model compilation fails.
    """
    # ── Define qubits ────────────────────────────────────────────────────────
    qubits = cirq.GridQubit.rect(1, n_qubits)  # 1-row grid of n_qubits qubits
    symbols = _make_symbols("θ", n_params)

    # ── Build the parameterised circuit ──────────────────────────────────────
    try:
        pqc_circuit = circuit_builder(qubits, symbols)
    except Exception as exc:
        raise ValueError(f"circuit_builder raised an error: {exc}") from exc

    if not isinstance(pqc_circuit, cirq.Circuit):
        raise ValueError(
            f"circuit_builder must return cirq.Circuit, got {type(pqc_circuit)}."
        )

    # ── Define observables: ⟨Z⟩ on every qubit ──────────────────────────────
    # Each qubit's Z-expectation value is in [-1, 1] and serves as a feature
    # for the downstream classical Dense layer.
    observables = [cirq.Z(q) for q in qubits]

    # ── Keras model ──────────────────────────────────────────────────────────
    circuit_input = tf.keras.Input(shape=(), dtype=tf.string, name="circuits")

    # tfq.layers.PQC resolves the symbols to obtain expectation values
    expectations = tfq.layers.PQC(
        pqc_circuit,
        observables,
        name="pqc_layer",
    )(circuit_input)

    # Classical readout: softmax over 10 MNIST classes
    logits = tf.keras.layers.Dense(n_classes, activation="softmax", name="readout")(
        expectations
    )

    model = tf.keras.Model(
        inputs=circuit_input, outputs=logits, name=circuit_builder.__name__
    )

    # ── Compile ───────────────────────────────────────────────────────────────
    try:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception as exc:
        raise RuntimeError(f"Model compilation failed: {exc}") from exc

    return model, symbols


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 – TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def train_model(
    model: tf.keras.Model,
    x_train: tf.Tensor,
    y_train: np.ndarray,
    x_val: tf.Tensor,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> tf.keras.callbacks.History:
    """
    Train a TFQ Keras model and return its training history.

    Args:
        model:      Compiled tf.keras.Model.
        x_train:    TFQ circuit tensor for training set.
        y_train:    Integer label array for training set.
        x_val:      TFQ circuit tensor for validation set.
        y_val:      Integer label array for validation set.
        epochs:     Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        Keras History object with loss/accuracy per epoch.

    Raises:
        RuntimeError: If training encounters a fatal error.
    """
    callbacks = [
        # Stop early if validation loss stops improving for 3 epochs
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=0
        ),
        # Halve LR when val_loss plateaus for 2 epochs
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, verbose=0
        ),
    ]

    try:
        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,  # suppress per-step output; we log ourselves
        )
    except Exception as exc:
        raise RuntimeError(f"Training failed: {exc}") from exc

    return history


def evaluate_model(
    model: tf.keras.Model,
    x_test: tf.Tensor,
    y_test: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a trained model and compute classification metrics.

    Metrics computed:
        • Balanced accuracy (accounts for class imbalance)
        • Macro precision   (equal weight per class)
        • Macro recall
        • Macro F1 score

    Args:
        model:  Trained tf.keras.Model.
        x_test: TFQ circuit tensor for test set.
        y_test: True integer labels.

    Returns:
        Dictionary with keys: balanced_accuracy, precision, recall, f1.

    Raises:
        ValueError: If model output shape is incompatible with y_test.
    """
    probs = model.predict(x_test, verbose=0)  # (N, n_classes)
    preds = np.argmax(probs, axis=1).astype(np.int32)

    if preds.shape != y_test.shape:
        raise ValueError(
            f"Prediction shape {preds.shape} ≠ label shape {y_test.shape}."
        )

    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
        "precision": float(
            precision_score(y_test, preds, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
    }
    return metrics, preds


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 – VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════


def _save(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to OUTPUT_DIR as PNG (300 dpi)."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {path}")


def plot_sample_digits(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    n_per_class: int = 5,
) -> None:
    """
    Display a grid of sample MNIST digits, one row per class.

    Args:
        x_raw:       Raw pixel arrays of shape (N, 28, 28).
        y_raw:       Integer labels of shape (N,).
        n_per_class: Number of sample images per digit class.
    """
    fig, axes = plt.subplots(
        N_CLASSES,
        n_per_class,
        figsize=(n_per_class * 1.5, N_CLASSES * 1.5),
    )
    fig.suptitle("MNIST Sample Digits (one row per class)", fontsize=14, y=1.01)

    for cls in range(N_CLASSES):
        cls_idx = np.where(y_raw == cls)[0][:n_per_class]
        for col, idx in enumerate(cls_idx):
            ax = axes[cls][col]
            ax.imshow(x_raw[idx], cmap="gray_r", interpolation="nearest")
            ax.axis("off")
        axes[cls][0].set_ylabel(str(cls), rotation=0, labelpad=15, fontsize=10)

    _save(fig, "01_sample_digits")


def plot_pca_variance(x_flat: np.ndarray, n_components: int = 30) -> None:
    """
    Plot explained variance ratio of PCA components to justify n_pca choice.

    Args:
        x_flat:       Flattened pixel array (N, 784).
        n_components: How many PCA components to analyse.
    """
    pca = PCA(n_components=n_components, random_state=SEED)
    pca.fit(x_flat)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Per-component bar chart
    ax1.bar(
        range(1, n_components + 1),
        pca.explained_variance_ratio_ * 100,
        color=sns.color_palette("Blues_d", n_components),
    )
    ax1.axvline(
        N_PCA_COMPONENTS,
        color="red",
        linestyle="--",
        label=f"Selected: {N_PCA_COMPONENTS}",
    )
    ax1.set_xlabel("PCA Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Per-Component Explained Variance")
    ax1.legend()

    # Cumulative curve
    ax2.plot(range(1, n_components + 1), cumvar, "b-o", markersize=4)
    ax2.axvline(
        N_PCA_COMPONENTS,
        color="red",
        linestyle="--",
        label=f"Selected ({N_PCA_COMPONENTS} comp.): {cumvar[N_PCA_COMPONENTS - 1]:.1f}%",
    )
    ax2.axhline(90, color="green", linestyle=":", label="90% threshold")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.legend()

    fig.suptitle("PCA Variance Analysis", fontsize=14)
    plt.tight_layout()
    _save(fig, "02_pca_variance")


def plot_circuit_diagrams(architectures: dict) -> None:
    """
    Print ASCII diagrams for each circuit architecture and save as text summary.

    cirq's built-in ASCII printer is used to render each circuit.

    Args:
        architectures: Dict mapping name → (circuit_builder, n_params_fn).
    """
    n_qubits = N_PCA_COMPONENTS
    qubits = cirq.GridQubit.rect(1, n_qubits)

    fig, axes = plt.subplots(
        len(architectures), 1, figsize=(14, 4 * len(architectures))
    )
    if len(architectures) == 1:
        axes = [axes]

    fig.suptitle("Quantum Circuit Architecture Diagrams", fontsize=14, y=1.01)

    for ax, (name, (builder, n_params_fn)) in zip(axes, architectures.items()):
        n_params = n_params_fn(n_qubits)
        symbols = _make_symbols("θ", n_params)
        circuit = builder(qubits, symbols)

        # Render cirq circuit as a string
        diagram = circuit.to_text_diagram(use_unicode_characters=False)

        ax.text(
            0.01,
            0.98,
            f"Architecture: {name}  │  Params: {n_params}",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.01,
            0.85,
            diagram,
            transform=ax.transAxes,
            fontsize=6.5,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="aliceblue", alpha=0.7),
        )
        ax.axis("off")

    plt.tight_layout()
    _save(fig, "03_circuit_diagrams")


def plot_training_curves(
    all_histories: dict[str, tf.keras.callbacks.History],
) -> None:
    """
    Plot training/validation loss and accuracy curves for all architectures.

    Args:
        all_histories: Dict mapping architecture name → Keras History.
    """
    n = len(all_histories)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    for col, (name, hist) in enumerate(all_histories.items()):
        epochs = range(1, len(hist.history["loss"]) + 1)

        # ── Loss ─────────────────────────────────────────────────────────────
        axes[0][col].plot(epochs, hist.history["loss"], "b-o", ms=4, label="Train")
        axes[0][col].plot(epochs, hist.history["val_loss"], "r--s", ms=4, label="Val")
        axes[0][col].set_title(f"{name}\nLoss", fontsize=9)
        axes[0][col].set_xlabel("Epoch")
        if col == 0:
            axes[0][col].set_ylabel("Loss")
        axes[0][col].legend(fontsize=7)
        axes[0][col].grid(alpha=0.3)

        # ── Accuracy ─────────────────────────────────────────────────────────
        axes[1][col].plot(epochs, hist.history["accuracy"], "b-o", ms=4, label="Train")
        axes[1][col].plot(
            epochs, hist.history["val_accuracy"], "r--s", ms=4, label="Val"
        )
        axes[1][col].set_title("Accuracy", fontsize=9)
        axes[1][col].set_xlabel("Epoch")
        axes[1][col].set_ylim(0, 1)
        if col == 0:
            axes[1][col].set_ylabel("Accuracy")
        axes[1][col].legend(fontsize=7)
        axes[1][col].grid(alpha=0.3)

    fig.suptitle("Training & Validation Curves", fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "04_training_curves")


def plot_confusion_matrices(
    all_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> None:
    """
    Plot normalised confusion matrices for all architectures.

    Args:
        all_preds: Dict mapping architecture name → predicted label array.
        y_true:    True label array.
    """
    n = len(all_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_true, preds, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=range(N_CLASSES),
            yticklabels=range(N_CLASSES),
            ax=ax,
            linewidths=0.3,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    fig.suptitle("Normalised Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "05_confusion_matrices")


def plot_metric_comparison(
    all_metrics: dict[str, dict[str, float]],
) -> None:
    """
    Bar chart comparing all four metrics across architectures.

    Args:
        all_metrics: Dict mapping architecture name → metrics dict.
    """
    metric_names = ["balanced_accuracy", "precision", "recall", "f1"]
    arch_names = list(all_metrics.keys())
    x = np.arange(len(arch_names))
    width = 0.18
    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (metric, colour) in enumerate(zip(metric_names, colours)):
        values = [all_metrics[arch][metric] for arch in arch_names]
        bars = ax.bar(
            x + i * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colour,
            alpha=0.85,
            edgecolor="white",
        )
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=45,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(
        [a.replace("build_", "").replace("_", "\n") for a in arch_names],
        fontsize=9,
    )
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Metric Comparison Across All Quantum Architectures", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "06_metric_comparison")


def plot_radar_chart(all_metrics: dict[str, dict[str, float]]) -> None:
    """
    Radar (spider) chart showing the four metrics for each architecture.

    Args:
        all_metrics: Dict mapping architecture name → metrics dict.
    """
    metric_labels = ["Balanced\nAccuracy", "Precision", "Recall", "F1 Macro"]
    n_metrics = len(metric_labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = sns.color_palette("husl", len(all_metrics))

    for (name, metrics), colour in zip(all_metrics.items(), colors):
        values = [
            metrics["balanced_accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        ]
        values += values[:1]  # close polygon

        label = name.replace("build_", "").replace("_", " ")
        ax.plot(angles, values, "o-", linewidth=2, color=colour, label=label)
        ax.fill(angles, values, alpha=0.1, color=colour)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Architecture Performance Radar", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(alpha=0.4)

    _save(fig, "07_radar_chart")


def plot_timing_comparison(timing: dict[str, float]) -> None:
    """
    Horizontal bar chart of training time per architecture.

    Args:
        timing: Dict mapping architecture name → training time in seconds.
    """
    names = [n.replace("build_", "").replace("_", "\n") for n in timing.keys()]
    times = list(timing.values())
    colors = sns.color_palette("muted", len(names))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.5)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{t:.1f}s",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Training Time per Architecture", fontsize=13)
    ax.grid(axis="x", alpha=0.3)

    _save(fig, "08_timing_comparison")


def plot_per_class_f1(
    all_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> None:
    """
    Heatmap of per-class F1 scores across architectures.

    Args:
        all_preds: Dict mapping architecture name → predicted label array.
        y_true:    True label array.
    """
    arch_names = list(all_preds.keys())
    f1_matrix = np.zeros((len(arch_names), N_CLASSES))

    for row, (name, preds) in enumerate(all_preds.items()):
        f1_per_class = f1_score(
            y_true,
            preds,
            average=None,
            labels=range(N_CLASSES),
            zero_division=0,
        )
        f1_matrix[row] = f1_per_class

    short_names = [n.replace("build_", "").replace("_", "\n") for n in arch_names]

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        f1_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=range(N_CLASSES),
        yticklabels=short_names,
        ax=ax,
        linewidths=0.3,
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Architecture")
    ax.set_title("Per-Class F1 Score by Architecture", fontsize=13)

    _save(fig, "09_per_class_f1")


def plot_final_summary_table(
    all_metrics: dict[str, dict[str, float]],
    timing: dict[str, float],
) -> None:
    """
    Render a styled summary table as an image.

    Args:
        all_metrics: Dict mapping architecture name → metrics dict.
        timing:      Dict mapping architecture name → training seconds.
    """
    arch_names = list(all_metrics.keys())
    short_names = [n.replace("build_", "").replace("_", " ") for n in arch_names]

    col_labels = [
        "Architecture",
        "Bal. Acc.",
        "Precision",
        "Recall",
        "F1 Macro",
        "Time (s)",
    ]
    rows = []
    for name, short in zip(arch_names, short_names):
        m = all_metrics[name]
        rows.append(
            [
                short,
                f"{m['balanced_accuracy']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                f"{timing[name]:.1f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 2.0)

    # Colour header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Highlight best F1 row
    best_idx = int(np.argmax([all_metrics[n]["f1"] for n in arch_names]))
    for j in range(len(col_labels)):
        table[best_idx + 1, j].set_facecolor("#D5E8D4")  # light green

    ax.set_title(
        "Final Comparison Table (green = best F1)",
        fontsize=13,
        pad=20,
    )
    _save(fig, "10_summary_table")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 – MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    End-to-end pipeline:
        1. Load and preprocess MNIST
        2. Encode data as TFQ circuits
        3. Build, train, and evaluate 5 quantum architectures
        4. Generate all visualisations
        5. Print comparative summary
    """
    log.info("══════════════════════════════════════════════")
    log.info("  TensorFlow-Quantum MNIST — 5 Architectures  ")
    log.info("══════════════════════════════════════════════")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    # Visualise raw digits and PCA variance before touching quantum circuits
    (x_tr_raw, y_tr_raw), _ = tf.keras.datasets.mnist.load_data()
    plot_sample_digits(x_tr_raw, y_tr_raw)
    x_flat = x_tr_raw[:5000].reshape(-1, 784).astype(np.float32) / 255.0
    plot_pca_variance(x_flat)

    # ── 2. Encode data as quantum circuits ────────────────────────────────────
    log.info("Encoding data as quantum circuits …")
    qubits = cirq.GridQubit.rect(1, N_PCA_COMPONENTS)

    x_train_circ = encode_data_as_circuits(x_train, qubits)
    x_test_circ = encode_data_as_circuits(x_test, qubits)

    # Split 10% of train → validation
    val_size = max(1, int(0.1 * len(x_train)))
    x_val_circ = x_train_circ[:val_size]
    y_val = y_train[:val_size]
    x_tr_circ = x_train_circ[val_size:]
    y_tr = y_train[val_size:]

    log.info(f"Splits │ train={len(y_tr)}, val={val_size}, test={len(y_test)}")

    # ── 3. Define architectures ───────────────────────────────────────────────
    architectures: dict[str, tuple] = {
        "build_basic_circuit": (build_basic_circuit, lambda n: _n_params_basic(n)),
        "build_entangled_circuit": (
            build_entangled_circuit,
            lambda n: _n_params_entangled(n),
        ),
        "build_layered_vqc": (build_layered_vqc, lambda n: _n_params_layered_vqc(n)),
        "build_hybrid_deep_qnn": (
            build_hybrid_deep_qnn,
            lambda n: _n_params_hybrid_deep(n),
        ),
        "build_ansatz_circuit": (build_ansatz_circuit, lambda n: _n_params_ansatz(n)),
    }

    # Plot circuit diagrams before training
    plot_circuit_diagrams(architectures)

    # ── 4. Train & evaluate all architectures ─────────────────────────────────
    all_histories: dict[str, tf.keras.callbacks.History] = {}
    all_metrics: dict[str, dict[str, float]] = {}
    all_preds: dict[str, np.ndarray] = {}
    timing: dict[str, float] = {}

    for arch_name, (builder, n_params_fn) in architectures.items():
        log.info(f"\n{'─' * 55}")
        log.info(f"  Training: {arch_name}")
        n_params = n_params_fn(N_PCA_COMPONENTS)
        log.info(f"  Qubits={N_PCA_COMPONENTS} │ Trainable params={n_params}")
        log.info(f"{'─' * 55}")

        try:
            model, _ = build_tfq_model(builder, N_PCA_COMPONENTS, n_params)
            model.summary(print_fn=lambda x: log.debug(x))  # log at DEBUG level

            t0 = time.time()
            history = train_model(model, x_tr_circ, y_tr, x_val_circ, y_val)
            elapsed = time.time() - t0
            timing[arch_name] = elapsed

            metrics, preds = evaluate_model(model, x_test_circ, y_test)
            all_histories[arch_name] = history
            all_metrics[arch_name] = metrics
            all_preds[arch_name] = preds

            log.info(
                f"  Done in {elapsed:.1f}s │ "
                f"Bal.Acc={metrics['balanced_accuracy']:.4f} │ "
                f"F1={metrics['f1']:.4f}"
            )

            # Per-architecture classification report
            log.info(
                "\n"
                + classification_report(
                    y_test,
                    preds,
                    target_names=[f"Digit {i}" for i in range(N_CLASSES)],
                    zero_division=0,
                )
            )

        except (ValueError, RuntimeError) as exc:
            # Log error but continue with remaining architectures
            log.error(f"  Architecture {arch_name} failed: {exc}")
            continue

    if not all_metrics:
        log.error("All architectures failed. Exiting.")
        sys.exit(1)

    # ── 5. Visualisations ─────────────────────────────────────────────────────
    log.info("\nGenerating visualisations …")
    plot_training_curves(all_histories)
    plot_confusion_matrices(all_preds, y_test)
    plot_metric_comparison(all_metrics)
    plot_radar_chart(all_metrics)
    plot_timing_comparison(timing)
    plot_per_class_f1(all_preds, y_test)
    plot_final_summary_table(all_metrics, timing)

    # ── 6. Console summary ────────────────────────────────────────────────────
    best_arch = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    log.info("\n" + "═" * 60)
    log.info("  FINAL SUMMARY")
    log.info("═" * 60)

    header = f"{'Architecture':<30} {'Bal.Acc':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Time':>7}"
    log.info(header)
    log.info("─" * len(header))
    for name, m in all_metrics.items():
        flag = " ← BEST" if name == best_arch else ""
        log.info(
            f"{name.replace('build_', ''):<30} "
            f"{m['balanced_accuracy']:>9.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{timing.get(name, 0):>6.1f}s"
            f"{flag}"
        )

    log.info("═" * 60)
    log.info(f"Best architecture (F1): {best_arch}")
    log.info(f"All plots saved to: {OUTPUT_DIR.resolve()}")
    log.info("Done.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
