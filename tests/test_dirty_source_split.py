import unittest

import pandas as pd

from scripts.generate_dirty_sources import split_customers


class DirtySourceSplitTests(unittest.TestCase):
    def test_split_covers_every_source_customer(self):
        source = pd.DataFrame(
            {
                "customer_id": [f"C{i:04d}" for i in range(1000)],
                "value": range(1000),
            }
        )

        app, web, overlap = split_customers(source)
        app_ids = set(app["customer_id"])
        web_ids = set(web["customer_id"])

        self.assertEqual(len(app_ids), 600)
        self.assertEqual(len(web_ids), 450)
        self.assertEqual(len(overlap), 50)
        self.assertEqual(app_ids | web_ids, set(source["customer_id"]))


if __name__ == "__main__":
    unittest.main()
