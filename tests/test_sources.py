from unittest import mock
import unittest

from deepsensor.data import sources


class TestDataSources(unittest.TestCase):
    """Tests for public data source helpers."""

    def test_era5_uses_anonymous_gcs_access(self):
        """ERA5 opens the public GCS dataset without authentication."""
        with mock.patch.object(
            sources.xr, "open_zarr", side_effect=RuntimeError("stop after open")
        ) as open_zarr:
            with self.assertRaisesRegex(RuntimeError, "stop after open"):
                sources._get_era5_reanalysis_data_parallel(
                    date_range=("2024-01-01", "2024-01-02"),
                    var_IDs=["2m_temperature"],
                    freq="H",
                    extent="global",
                    cache=False,
                )

        args, kwargs = open_zarr.call_args
        self.assertTrue(args[0].startswith("gs://gcp-public-data-arco-era5/"))
        self.assertEqual(kwargs["storage_options"], {"token": "anon"})


if __name__ == "__main__":
    unittest.main()
