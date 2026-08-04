"""Central code — basic objects (PDF item 8).

Phase 1.A: bundle context + functions + vector fields with the
``U(f)`` action. Later sub-phases added forms, p-vectors, tensors,
metric, connection, frame, interior/wedge, etc.
"""

from jacopy.central.objects.bundle import (
    Bundle,
    TangentBundle,
    tangent_bundle,
    TM,
)
from jacopy.central.objects.function import functions
from jacopy.central.objects.vector_field import (
    VectorField,
    vector_fields,
    apply_to_function,
)
from jacopy.central.objects.form import (
    Form,
    forms,
    form_on_vector,
    vector_on_form,
)
from jacopy.central.objects.multivector import (
    PVector,
    p_vectors,
    bivector,
)
from jacopy.central.objects.tensor import (
    Tensor,
    tensors,
    scale,
    signature_of,
)
from jacopy.central.objects.interior import (
    Interior,
    interior,
    contract,
    contract_all,
)
from jacopy.central.objects.tilde_interior import (
    TildeInterior,
    tilde_interior,
    tilde_contract,
)
from jacopy.central.objects.musical import (
    Flat,
    Sharp,
    flat,
    sharp,
)
from jacopy.central.objects.metric import (
    Metric,
    InverseMetric,
    metric,
    hodge,
)
from jacopy.central.objects.connection import (
    Connection,
    CovariantOp,
    connection,
    covariant_derivative,
)
from jacopy.central.objects.frame import (
    Frame,
    Coframe,
    FrameField,
    CoframeField,
    KroneckerDelta,
    frame,
    coframe,
    kronecker_delta,
)
from jacopy.core.hodge import HodgeStar, hodge_star
from jacopy.core.pairing import Pairing, pairing
from jacopy.core.multi_eval import MultiEval, multi_eval
from jacopy.core.wedge import Wedge, wedge
from jacopy.core.tensor_product import TensorProduct, tensor_product
from jacopy.core.symmetrize import (
    Symmetrization,
    Antisymmetrization,
    symmetrize,
    antisymmetrize,
)

__all__ = [
    # bundle
    "Bundle",
    "TangentBundle",
    "tangent_bundle",
    "TM",
    # functions
    "functions",
    # vector fields
    "VectorField",
    "vector_fields",
    "apply_to_function",
    # forms + pairing + evaluation
    "Form",
    "forms",
    "form_on_vector",
    "vector_on_form",
    "Pairing",
    "pairing",
    "MultiEval",
    "multi_eval",
    # interior
    "Interior",
    "interior",
    "contract",
    "contract_all",
    # tilde-interior
    "TildeInterior",
    "tilde_interior",
    "tilde_contract",
    # musical ♯/♭
    "Flat",
    "Sharp",
    "flat",
    "sharp",
    # multivectors
    "PVector",
    "p_vectors",
    "bivector",
    # tensors + fT + signature tracking
    "Tensor",
    "tensors",
    "scale",
    "signature_of",
    # products + (anti)symmetrization
    "Wedge",
    "wedge",
    "TensorProduct",
    "tensor_product",
    "Symmetrization",
    "Antisymmetrization",
    "symmetrize",
    "antisymmetrize",
    # metric + inverse + hodge
    "Metric",
    "InverseMetric",
    "metric",
    "hodge",
    "HodgeStar",
    "hodge_star",
    # connection
    "Connection",
    "CovariantOp",
    "connection",
    "covariant_derivative",
    # frame + coframe
    "Frame",
    "Coframe",
    "FrameField",
    "CoframeField",
    "frame",
    "coframe",
    "kronecker_delta",
    "KroneckerDelta",
]
