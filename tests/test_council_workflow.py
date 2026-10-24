import json
import unittest
from copy import deepcopy

from services.config_store import DEFAULT_COUNCIL_CONFIG, normalize_council
from services.council_workflow import CouncilStopped, parse_ballot, run_council


class CouncilWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.settings = normalize_council(deepcopy(DEFAULT_COUNCIL_CONFIG))
        self.calls = []
        self.records = []
        self.votes = {}
        self.more = False
        self.failure = None

    def request(self, member, system, prompt, phase, number, private):
        self.calls.append((member['id'], phase, number, json.loads(prompt), private))
        if self.failure:
            failure = self.failure(member, phase, number)
            if failure:
                raise failure
        if phase == 'proposal':
            return f"Independent solution from {member['id']}"
        if phase == 'discussion':
            return json.dumps({'contribution': f"Useful idea from {member['id']}",
                               'needs_more_discussion': self.more})
        if phase == 'draft':
            return f'Exact draft {number}\n\nWith evidence and dissent.'
        if phase == 'vote':
            # No ballot is published until every seat has voted on this draft.
            self.assertFalse(any(record[0] == 'vote' and record[1] == number for record in self.records))
            return json.dumps(self.votes.get((member['id'], number),
                                            {'vote': 'approve', 'reason': 'Reviewed evidence.'}))
        raise AssertionError(phase)

    def run_workflow(self, **kwargs):
        return run_council(settings=self.settings, user_message='Solve the whole problem',
                           request_turn=self.request, publish=lambda *args: self.records.append(args), **kwargs)

    def test_order_independence_private_ballots_and_exact_approved_draft(self):
        result = self.run_workflow()
        ids = [member['id'] for member in self.settings['members']]
        self.assertEqual([(m, p) for m, p, *_ in self.calls],
                         [(m, 'proposal') for m in ids] + [(m, 'discussion') for m in ids]
                         + [('chair', 'draft')] + [(m, 'vote') for m in ids if m != 'chair'])
        for _, phase, _, context, private in self.calls:
            if phase == 'proposal':
                self.assertEqual(context['council_record'], [])
            if phase == 'vote':
                self.assertTrue(private)
                self.assertFalse(any(record['phase'] == 'ballots' for record in context['council_record']))
                self.assertEqual(context['council_record'][-1]['text'], result.draft)
        self.assertTrue(result.approved)
        self.assertEqual(result.draft, 'Exact draft 1\n\nWith evidence and dissent.')
        self.assertIn(result.draft, result.markdown())
        self.assertEqual(self.calls[-1][1], 'vote')

    def test_tie_requires_revision_then_whole_electorate_votes_again(self):
        self.votes[('member-2', 1)] = {'vote': 'request_changes', 'reason': 'Missing a constraint.'}
        result = self.run_workflow()
        self.assertTrue(result.approved)
        self.assertEqual(result.draft_number, 2)
        revisions = [call for call in self.calls if call[1] == 'draft']
        self.assertIn('Missing a constraint.', json.dumps(revisions[-1][3]))
        self.assertEqual(len([call for call in self.calls if call[1] == 'vote']), 4)

    def test_missing_voter_does_not_reduce_majority_threshold(self):
        self.settings['max_revisions'] = 0
        self.failure = lambda m, p, n: TimeoutError('slow') if m['id'] == 'member-2' and p == 'vote' else None
        result = self.run_workflow()
        self.assertFalse(result.approved)
        self.assertEqual([ballot['vote'] for ballot in result.ballots], ['approve', 'unavailable'])
        self.assertIn('Not approved', result.markdown())

    def test_invalid_or_abstaining_votes_do_not_approve(self):
        for ballot in ({'vote': 'abstain', 'reason': 'No basis.'},
                       {'vote': 'approve', 'reason': ''}, {'vote': 'maybe', 'reason': 'Unsure'}):
            with self.subTest(ballot=ballot):
                self.calls.clear(); self.records.clear()
                self.settings['max_revisions'] = 0
                self.votes[('member-2', 1)] = ballot
                self.assertFalse(self.run_workflow().approved)

    def test_round_and_revision_limits(self):
        self.more = True
        for member in self.settings['members'][1:]:
            for number in (1, 2):
                self.votes[(member['id'], number)] = {'vote': 'reject', 'reason': 'Fundamental issue.'}
        result = self.run_workflow()
        self.assertFalse(result.approved)
        self.assertEqual(len([call for call in self.calls if call[1] == 'discussion']), 6)
        self.assertEqual(len([call for call in self.calls if call[1] == 'draft']), 2)

    def test_stop_never_becomes_failure_or_starts_another_turn(self):
        self.failure = lambda m, p, n: CouncilStopped() if p == 'proposal' else None
        with self.assertRaises(CouncilStopped):
            self.run_workflow()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.records, [])

    def test_failed_proposal_is_not_used_as_evidence(self):
        self.failure = lambda m, p, n: RuntimeError('offline') if m['id'] == 'member-1' and p == 'proposal' else None
        self.assertTrue(self.run_workflow().approved)
        discussion = next(call for call in self.calls if call[1] == 'discussion')
        self.assertEqual(discussion[3]['council_record'][1]['text'], '[Turn failed]')

    def test_chair_failure_does_not_vote_on_incomplete_draft(self):
        self.failure = lambda m, p, n: RuntimeError('offline') if p == 'draft' else None
        result = self.run_workflow()
        self.assertFalse(result.approved)
        self.assertEqual(result.draft_number, 0)
        self.assertFalse(any(call[1] == 'vote' for call in self.calls))

    def test_updates_are_frozen_until_boundary(self):
        updates = []
        original = self.request
        def request(*args):
            if args[3] == 'proposal':
                updates.append('New constraint')
            return original(*args)
        result = run_council(settings=self.settings, user_message='Q', request_turn=request,
                             publish=lambda *args: self.records.append(args),
                             take_redirects=lambda: [updates.pop(0) for _ in range(len(updates))])
        proposals = [call for call in self.calls if call[1] == 'proposal']
        self.assertTrue(all(call[3]['user_updates'] == [] for call in proposals))
        discussion = next(call for call in self.calls if call[1] == 'discussion')
        self.assertEqual(len(discussion[3]['user_updates']), 3)
        self.assertTrue(result.approved)

    def test_update_during_vote_invalidates_approval(self):
        updates = []
        self.settings['max_revisions'] = 0
        original = self.request
        def request(*args):
            if args[3] == 'vote':
                updates.append('Changed goal')
            return original(*args)
        result = run_council(settings=self.settings, user_message='Q', request_turn=request,
                             publish=lambda *args: self.records.append(args),
                             take_redirects=lambda: [updates.pop(0) for _ in range(len(updates))])
        self.assertFalse(result.approved)
        self.assertIn('new ballot', result.error)
        self.assertFalse(next(record[4]['approved'] for record in self.records if record[0] == 'tally'))

    def test_fenced_ballot_is_parsed_but_free_text_is_not_a_vote(self):
        self.assertEqual(parse_ballot('```json\n{"vote":"approve","reason":"OK"}\n```')['vote'], 'approve')
        self.assertEqual(parse_ballot('I approve')['vote'], 'invalid')


if __name__ == '__main__':
    unittest.main()
