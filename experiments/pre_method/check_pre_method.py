"""CPU checks for the pre-method code; no data, no encoders, no GPU.

    python experiments/pre_method/check_pre_method.py
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

import common  # noqa: E402
from e0_geometry_audit import all_geodesic, matched_geodesic  # noqa: E402
from pilot.geometry import EuclideanSpace, LorentzSpace, gromov  # noqa: E402
from pilot.model import SubsetEncoder, all_subsets  # noqa: E402
from pilot.objectives import Objective  # noqa: E402
from utils import lorentz_spectral as lorentz  # noqa: E402


class Statistics(unittest.TestCase):
    def test_rank_ties_and_spearman(self):
        self.assertEqual(common.rankdata([3, 1, 1, 2]).tolist(), [3.0, 0.5, 0.5, 2.0])
        x = torch.randn(200, dtype=torch.float64)
        self.assertAlmostEqual(common.spearman(x, x.exp()), 1.0, places=12)
        self.assertAlmostEqual(common.spearman(x, -x), -1.0, places=12)

    def test_recall(self):
        scores = torch.eye(5) + 0.1 * torch.arange(5.)[None]
        self.assertEqual(common.recall_at_k(scores, torch.arange(5))["R@1"], 100.0)

    def test_load_dump_merges_rank_shards_and_drops_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            for rank, ids in enumerate((["a", "b"], ["b", "c"])):
                torch.save({"meta": {"dataset": "x"}, "ids": ids, "captions": ids,
                            "proj": {"t": torch.tensor([[float(ord(i))] for i in ids])}},
                           common.shard_path(d, "x", rank))
            dump = common.load_dump(d, "x")
            self.assertEqual(dump["ids"], ["a", "b", "c"])
            self.assertEqual(dump["proj"]["t"].squeeze(1).tolist(), [97.0, 98.0, 99.0])
            self.assertEqual(common.list_datasets(d), ["x"])


class Geometry(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.lorentz = LorentzSpace(curvature_init=0.7)
        self.euclid = EuclideanSpace()

    def points(self, space, *shape, scale=0.8):
        return space.embed(torch.randn(*shape, dtype=torch.float64) * scale)

    def test_radius_and_log_map_inverse_expmap(self):
        tangent = torch.randn(50, 6, dtype=torch.float64)
        x = self.lorentz.embed(tangent)
        c = self.lorentz.c.item()
        self.assertTrue(torch.allclose(common.lorentz_radius(x, c), tangent.norm(dim=-1), atol=1e-9))
        self.assertTrue(torch.allclose(common.lorentz_log0(x, c), tangent, atol=1e-9))

    def test_distances_agree_and_dominate_tangent_distance(self):
        x, y = self.points(self.lorentz, 40, 5), self.points(self.lorentz, 40, 5)
        c = self.lorentz.c.item()
        pair = self.lorentz.distance(x, y)
        self.assertTrue(torch.allclose(pair.diagonal(), self.lorentz.matched_distance(x, y), atol=1e-9))
        self.assertTrue(torch.allclose(pair, all_geodesic(x, y, c), atol=1e-9))
        self.assertTrue(torch.allclose(pair.diagonal(), matched_geodesic(x, y, c), atol=1e-9))
        flat = (common.lorentz_log0(x, c) - common.lorentz_log0(y, c)).norm(dim=-1)
        self.assertTrue((self.lorentz.matched_distance(x, y) >= flat - 1e-9).all())  # CAT(0) comparison

    def test_gromov_product_bounds(self):
        for space in (self.lorentz, self.euclid):
            x, y = self.points(space, 30, 4), self.points(space, 30, 4)
            g = gromov(space, x, y, matched=True)
            self.assertTrue((g >= -1e-9).all())
            self.assertTrue((g <= torch.minimum(space.radius(x), space.radius(y)) + 1e-9).all())

    def test_cones_contain_the_outward_ray_and_reject_the_inward_one(self):
        direction = torch.nn.functional.normalize(torch.randn(20, 4, dtype=torch.float64), dim=-1)
        for space in (self.lorentz, self.euclid):
            parent = space.embed(direction * 1.0)
            outward = space.embed(direction * 2.0)
            inward = space.embed(direction * 0.5)
            self.assertTrue((space.cone_violation(parent, outward) < 1e-3).all(), space.name)
            self.assertTrue((space.cone_violation(parent, inward) > 1.0).all(), space.name)

    def test_centroids_are_on_the_manifold_and_inside(self):
        points = self.points(self.lorentz, 25, 3, 5, scale=1.5)
        c = self.lorentz.c
        for perron in (False, True):
            centre = self.lorentz.centroid(points, perron=perron)
            time = lorentz.time_component(centre, c).squeeze(-1)
            self.assertTrue(torch.allclose(c * (time.square() - centre.square().sum(-1)),
                                           torch.ones(25, dtype=torch.float64), atol=1e-9))
            self.assertTrue((self.lorentz.radius(centre) <= self.lorentz.radius(points).max(-1).values + 1e-9).all())


class Pilot(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.dims = {"t": 12, "v": 20, "a": 8}
        self.features = {m: torch.randn(16, d) for m, d in self.dims.items()}

    def model(self, space, encoder="fusion"):
        return SubsetEncoder(self.dims, "tva", space, 8, encoder, d_model=32, layers=1, heads=4, dropout=0.0)

    def test_every_objective_backpropagates_in_both_spaces(self):
        for space_cls in (EuclideanSpace, LorentzSpace):
            for name in ("allpair", "collapse", "subsets", "order"):
                for text_mode in ("lattice", "query"):
                    if name == "collapse" and text_mode == "query":
                        continue
                    model = self.model(space_cls())
                    objective = Objective(name, "tva", text_mode=text_mode, pairs_per_step=3)
                    loss, logs = objective(model, self.features, torch.tensor(10.0))
                    loss.backward()
                    self.assertTrue(math.isfinite(loss.item()), (space_cls.name, name))
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads), (space_cls.name, name))

    def test_subset_encoders_and_covering_pairs(self):
        self.assertEqual(len(all_subsets("tva")), 7)
        self.assertEqual(len(Objective("order", "tva").covering), 9)
        self.assertEqual(len(Objective("order", "tva", text_mode="query").subsets), 3)
        for encoder in ("fusion", "centroid", "perron"):
            model = self.model(LorentzSpace(), encoder)
            out = model(self.features, ("t", "v", "a"))
            self.assertEqual(out.shape, (16, 8))
            self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
