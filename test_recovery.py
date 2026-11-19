import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
import daily

class Recovery(unittest.TestCase):
    def test_worker_checkpoint_returns_unprocessed_codes_for_recovery(self):
        from contextlib import chdir
        import tempfile
        with tempfile.TemporaryDirectory() as root,chdir(root), \
             patch.object(daily,'monotonic',side_effect=[0,3001]), \
             patch.object(daily.bs,'query_history_k_data_plus') as query:
            frames,failed=daily.fetch_shard(1,['sh.600001','sz.000001'],'2025-01-01','2026-11-19')
            self.assertFalse(frames)
            self.assertEqual([code for code,_ in failed],['sh.600001','sz.000001'])
            query.assert_not_called()

    def test_premarket_defaults_to_last_closed_session(self):
        clock=datetime(2026,11,19,12,0,tzinfo=ZoneInfo('Asia/Shanghai'))
        with patch('sys.argv',['daily.py']),patch.object(daily,'datetime') as time, \
             patch.object(daily.bs,'login',return_value=SimpleNamespace(error_code='0')), \
             patch.object(daily.bs,'logout'),patch.object(daily,'run') as run:
            time.now.return_value=clock
            daily.main()
            run.assert_called_once_with('2026-11-18')

    def test_zero_tradestatus_falls_back_to_dated_security_identity(self):
        basic=pd.DataFrame({'code':['sh.600001','sz.000001','sh.600002'],
            'code_name':['正常','已退市','未上市'],'ipoDate':['2000-01-01','2000-01-01','2027-01-01'],
            'outDate':['','2026-01-01','']})
        with patch.object(daily.bs,'query_all_stock'),patch.object(daily.bs,'query_stock_basic'), \
             patch.object(daily,'read',side_effect=[pd.DataFrame()]*3+[basic]),patch.object(daily,'sleep'):
            result=daily.load_universe('2026-11-19')
            self.assertEqual(result.code.tolist(),['sh.600001'])

if __name__=='__main__':unittest.main()
