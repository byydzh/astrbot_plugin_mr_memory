import asyncio
import json
import shutil
import uuid
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mr_memory.store import Store
from mr_memory.agent import MemoryAgent
from mr_memory.memory_protocol import tool_definitions
from mr_memory.settings import normalize_settings


class MemberProfileTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.dev'
        self.directory = self.root / ('test-members-' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.store = Store(self.directory / 'group.db', 'test:GroupMessage:42')
        self.now = int(time.time())

    def tearDown(self):
        self.store.close()
        self.assertEqual(self.directory.resolve().parent,self.root.resolve())
        shutil.rmtree(self.directory)

    def say(self, uid, suffix='', at=None, name=None):
        return self.store.append_message(dict(platform_id='test', sender_id=str(uid),
            message_id=f'{uid}-{suffix}', sender_name=name or f'成员{uid}', sent_at=at or self.now,
            plain_text='测试发言'+suffix, content=[], role='USER'))

    def test_full_directory_and_daily_stable_pool_are_separate(self):
        for i in range(60):
            self.say(i)
        self.store.cache_roster([{'user_id':str(i),'nickname':f'昵称{i}','card':f'名片{i}'} for i in range(70)], self.now)
        first = self.store.member_pool(50, now=self.now)
        self.assertEqual(len(first['account_ids']),50)
        self.assertEqual(len(self.store.members()),70)
        excluded = next(str(i) for i in range(60) if str(i) not in first['account_ids'])
        for j in range(5):
            self.say(excluded,str(j))
        self.assertEqual(self.store.member_pool(50,now=self.now)['prefix'],first['prefix'])
        tomorrow = self.store.member_pool(50,now=self.now+86400)
        self.assertIn(excluded,tomorrow['account_ids'])
        self.assertEqual(tomorrow['account_ids'],sorted(tomorrow['account_ids']))
        self.assertEqual(len(self.store.member_pool(3,now=self.now+86400)['account_ids']),3)
        self.assertEqual(len(self.store.members()),70)
        self.assertTrue(self.store.members(['69']))
        stable = self.store.member_pool(50,now=self.now+86400)['prefix']
        self.store.cache_roster([{'user_id':str(i),'nickname':f'改名{i}','card':f'名片{i}'} for i in range(70)],self.now+86401)
        self.assertEqual(self.store.member_pool(50,now=self.now+86402)['prefix'],stable)
        self.assertEqual(self.store.members(['69'])[0]['nickname'],'改名69')

    def test_self_information_survives_platform_changes_and_same_names(self):
        self.say('a',name='重名')
        self.say('b',name='重名')
        self.store.cache_roster([{'user_id':'a','nickname':'昵称','card':'名片'}],self.now)
        self.store.member_pool(50,now=self.now)
        self.store.edit_member('a',{'preferred_name':'新称呼','aliases':['小甲'], 'avoided_names':['旧称呼']},actor='self:a')
        self.store.cache_roster([{'user_id':'a','nickname':'新昵称','card':'新名片'}],self.now+1)
        a=self.store.members(['a'])[0]
        self.assertEqual(a['preferred_name'],'新称呼')
        self.assertEqual(a['avoided_names'],['旧称呼'])
        self.assertIn('昵称',a['observed_names'])
        self.assertEqual(a['card'],'新名片')
        self.assertEqual(len(self.store.members(name='重名')),2)
        self.assertEqual(self.store.members(['b'])[0]['confirmed_aliases'],[])
        self.assertIn('新称呼',self.store.member_pool(50,now=self.now+1)['prefix'])
        self.store.close()
        self.store=Store(self.directory/'group.db','test:GroupMessage:42')
        self.assertEqual(self.store.members(['a'])[0]['preferred_name'],'新称呼')

    def test_basic_mode_defaults_and_new_raw_semantic_index(self):
        settings=normalize_settings({})
        self.assertFalse(settings['advanced_memory_enabled'])
        self.assertFalse(settings['feedback_learning_enabled'])
        self.assertTrue(settings['subconscious_enabled'])
        self.assertTrue(settings['auto_distillation_enabled'])
        self.assertEqual(settings['member_prefix_capacity'],50)
        self.assertNotIn('remember',tool_definitions(basic=True))
        self.assertNotIn('reflect',tool_definitions(basic=True))
        self.assertIn('navigate',tool_definitions(basic=True))
        self.assertIn('remember',tool_definitions(basic=True,learning=True))
        self.assertNotIn('reflect',tool_definitions(basic=True,learning=True))
        self.assertEqual(tool_definitions(basic=True)['complete']['parameters']['properties']['background']['type'],'string')
        row=self.say('a')
        docs=self.store.pending_embeddings('test-vector')
        doc=next(d for d in docs if d['owner_type']=='message')
        self.assertEqual(int(doc['owner_key']),row['id'])
        self.assertTrue(self.store.save_embedding(doc,'test-vector',[1.,0.]))
        self.assertEqual(len(self.store.vector_rows('test-vector')),1)
        self.assertEqual(self.store.memory('message',row['id'])['sources'][0]['sender_id'],'a')


class BasicRecallTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_preserved_and_no_cognition_or_write(self):
        class Provider:
            provider_config={}
            calls=[]
            async def _prepare_chat_payload(self,**kwargs):
                return {'messages':[{'role':'system','content':kwargs['system_prompt']},*kwargs['contexts']]},None
            async def _query(self,payload,tools,**kwargs):
                self.calls.append(payload)
                return SimpleNamespace(completion_text=json.dumps({'background':'群友刚换了显示名，UID没有变化。',
                    'items':[{'kind':'semantic','content':'不应写入'}]}),tools_call_name=[],usage={})
        provider=Provider()
        agent=MemoryAgent(provider,object())
        agent.member_prefix='稳定的成员资料'
        with patch('mr_memory.agent._tool_set',return_value=object()):
            result=await agent.reconstruct({'sent_at':100},[],{'old':'不得带入旧理解'})
        self.assertEqual(result.status,'completed')
        self.assertEqual(result.written,[])
        self.assertIn('稳定的成员资料',provider.calls[0]['messages'][0]['content'])
        self.assertNotIn('不得带入旧理解',str(provider.calls))


if __name__ == '__main__':
    unittest.main()
