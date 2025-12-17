from __future__ import annotations

import SimpleITK as sitk

from Preprocessing.elastix_resample import transform_from_elastix_parameter_maps


def test_f2a_translation_only_parameter_maps_produce_3dof_transform() -> None:
    pm = sitk.ParameterMap()
    pm["Transform"] = ["TranslationTransform"]
    pm["TransformParameters"] = ["1.0", "2.0", "3.0"]

    vec = sitk.VectorOfParameterMap()
    vec.append(pm)

    t = transform_from_elastix_parameter_maps(vec)
    # CompositeTransform wraps a single translation; the contained transform must be 3 DOF.
    assert t.GetNumberOfTransforms() == 1
    inner = t.GetNthTransform(0)
    assert len(inner.GetParameters()) == 3
