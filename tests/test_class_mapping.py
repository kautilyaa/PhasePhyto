"""Tests for PlantDoc -> PlantVillage class-name mapping helpers."""


from phasephyto.data.class_mapping import mapped_plantdoc_overlap


def test_mapped_plantdoc_overlap_matches_manifest_example() -> None:
    source_counts = {
        "Apple___Apple_scab": 10,
        "Corn_(maize)___Common_rust_": 20,
        "Potato___Early_blight": 30,
        "Soybean___healthy": 40,
    }
    target_counts = {
        "Apple Scab Leaf": 1,
        "Corn rust leaf": 2,
        "Potato leaf early blight": 3,
        "Soyabean leaf": 4,
        "Tomato leaf": 5,
    }

    rows = mapped_plantdoc_overlap(source_counts, target_counts)

    assert len(rows) == 4
    assert {row["target"] for row in rows} == {
        "Apple Scab Leaf",
        "Corn rust leaf",
        "Potato leaf early blight",
        "Soyabean leaf",
    }
    assert {row["source"] for row in rows} == set(source_counts)
