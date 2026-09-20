"""Focused AstrBot hooks: self-only editing and both-model member delivery."""
import unittest
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from astrbot.api.provider import ProviderRequest
from tests import test_main as fixtures
from mr_memory.agent import ReconstructionResult


class MemberHookTests(unittest.IsolatedAsyncioTestCase):
    mode_settings = {}
    asyncSetUp = fixtures.MainIntegrationTests.asyncSetUp
    asyncTearDown = fixtures.MainIntegrationTests.asyncTearDown

    async def test_self_command_and_directory_handoff(self):
        current=fixtures.event()
        current.message_obj.timestamp=int(time.time())
        current.message_obj.raw_message={'time':current.message_obj.timestamp}
        current.bot=SimpleNamespace(get_group_member_list=AsyncMock(return_value=[
            {'user_id':current.get_sender_id(),'nickname':'平台名','card':'群名片'},
            {'user_id':'other','nickname':'另一个人','card':'群名片'}]))
        current.message_str='/mr uid 称呼 自选称呼'
        current.message_obj.message_str=current.message_str
        replies=[reply async for reply in self.plugin.member_profile(current)]
        self.assertEqual(len(replies),1)
        store=await self.plugin.store_for(current)
        self.assertEqual(store.members([current.get_sender_id()])[0]['preferred_name'],'自选称呼')
        self.assertEqual(store.members(['other'])[0]['preferred_name'],'')
        current.message_str='/mr uid other 称呼 越权改名'
        current.message_obj.message_str=current.message_str
        _=[reply async for reply in self.plugin.member_profile(current)]
        self.assertEqual(store.members(['other'])[0]['preferred_name'],'')
        self.assertFalse(self.plugin.config['advanced_memory_enabled'])
        agent=SimpleNamespace(reconstruct=AsyncMock(return_value=ReconstructionResult(background='可用背景',status='completed')))
        req=ProviderRequest(prompt='你记得我吗',system_prompt='原人格',contexts=[{'role':'user','content':'已有历史'}])
        with patch.object(self.plugin,'agent',return_value=agent):
            await self.plugin.inject_subconscious_memory(current,req)
        self.assertIn('自选称呼',agent.member_prefix)
        self.assertIn(agent.member_prefix,req.system_prompt)
        self.assertTrue(req.system_prompt.startswith('原人格'))
        self.assertEqual(req.contexts,[{'role':'user','content':'已有历史'}])
        self.assertIn('自选称呼',next(p.text for p in req.extra_user_content_parts if p.text.startswith('<mr_current_members>')))


if __name__=='__main__':
    unittest.main()
