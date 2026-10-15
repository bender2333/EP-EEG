"""Tests that the dataset boundary cannot silently widen to session E."""

from __future__ import annotations

import importlib
import unittest
from unittest.mock import patch


loader = importlib.import_module("eval.t_only_bnci")


class TOnlyLoaderTests(unittest.TestCase):
    def test_requests_only_training_file(self) -> None:
        calls: list[str] = []

        def fake_data_path(url: str, **_: object) -> list[str]:
            calls.append(url)
            return ["A01T.mat"]

        def fake_convert(*_: object, **__: object):
            return ["run0", "run1"], dict(loader.EVENTS)

        with patch.object(loader, "data_path", fake_data_path), patch.object(
            loader, "_convert_mi", fake_convert
        ):
            sessions, events = loader.load_t_session(1)

        self.assertEqual(calls, ["https://lampx.tugraz.at/~bci/database/001-2014/A01T.mat"])
        self.assertEqual(set(sessions), {"0train"})
        self.assertNotIn("E", calls[0])
        self.assertEqual(events, loader.EVENTS)

    def test_rejects_invalid_subject(self) -> None:
        with self.assertRaisesRegex(ValueError, "1 through 9"):
            loader.load_t_session(0)

    def test_rejects_unexpected_event_mapping(self) -> None:
        with patch.object(loader, "data_path", lambda *_args, **_kwargs: ["A01T.mat"]), patch.object(
            loader, "_convert_mi", lambda *_args, **_kwargs: (["run0"], {"wrong": 1})
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected event mapping"):
                loader.load_t_session(1)


if __name__ == "__main__":
    unittest.main()
