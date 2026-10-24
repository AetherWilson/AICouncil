import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from services.config_store import ConfigStore, DEFAULT_COUNCIL_CONFIG, normalize_council


class CouncilConfigTests(unittest.TestCase):
    def test_explicit_order_chair_and_disabled_seats(self):
        config = deepcopy(DEFAULT_COUNCIL_CONFIG)
        council = config['council']
        council['members'] = [council['members'][2], council['members'][0], council['members'][1]]
        council['members'][2]['enabled'] = False
        self.assertEqual([m['id'] for m in normalize_council(config)['members']], ['member-2', 'chair'])

    def test_legacy_models_are_seats_not_skills_and_no_reader_is_created(self):
        config = {'Leader': 'my-chair', 'skills': {'model_map': {'researcher-skill': 'other-model',
                                                              'pdfer-skill': 'reader-model'}},
                  'memory': {'auto_extract': True}, 'lite_model': 'helper'}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text(json.dumps(config))
            loaded = ConfigStore(folder).load_config()
        self.assertEqual([m['model'] for m in loaded['council']['members']], ['my-chair', 'other-model'])
        self.assertFalse(loaded['memory']['auto_extract'])
        self.assertNotIn('skills', loaded)
        self.assertNotIn('lite_model', loaded)

    def test_invalid_seats_do_not_silently_change_the_electorate(self):
        for case in ('duplicate', 'missing_chair', 'one_seat', 'missing_model', 'unsafe_id'):
            with self.subTest(case=case):
                config = deepcopy(DEFAULT_COUNCIL_CONFIG)
                seats = config['council']['members']
                if case == 'duplicate': seats[1]['id'] = seats[0]['id']
                if case == 'missing_chair': config['council']['chair'] = 'missing'
                if case == 'one_seat': config['council']['members'] = seats[:1]
                if case == 'missing_model': seats[1]['model'] = ''
                if case == 'unsafe_id': seats[1]['id'] = 'seat" onclick="x'
                with self.assertRaises(ValueError): normalize_council(config)

    def test_malformed_config_does_not_silently_replace_electorate(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, 'config.json').write_text('{broken')
            with self.assertRaises(ValueError):
                ConfigStore(folder).load_config()

    def test_limits_are_bounded(self):
        config = deepcopy(DEFAULT_COUNCIL_CONFIG)
        config['council'].update(discussion_rounds=-1, max_revisions=1000, turn_timeout_seconds='bad')
        settings = normalize_council(config)
        self.assertEqual((settings['discussion_rounds'], settings['max_revisions'], settings['turn_timeout_seconds']), (1, 5, 120))


if __name__ == '__main__':
    unittest.main()
