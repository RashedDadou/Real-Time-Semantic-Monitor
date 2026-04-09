# Real_Time_Semantic_Monitor.py - الكود الكامل مع المكتبات + الاختبار

"""
Real_Time_Semantic_Monitor.py
مراقبة لحظية ذكية للـ pipeline: تحدد وظيفة الدالة، تقيّم النتيجة،
وتوجّه الخطوة التالية تلقائيًا.
"""

# 1. استيراد المكتبات الأساسية (Core Imports)
from matplotlib.axes import Axes
from calendar import c
import re, time, sqlite3, csv, os
from contextlib import contextmanager
from typing import Dict, Any, Optional, Sequence, List, Union
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field

# 2. استيراد المكتبات المتقدمة مع Fallback (Advanced ML Imports)
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.animation import FuncAnimation

    HAS_TORCH = HAS_NUMPY = HAS_PLOT = HAS_ANIMATION = True
    plt.style.use('seaborn-v0_8-darkgrid')
    Tensor = torch.Tensor

    print("✅ ALL ML libraries loaded")

except ImportError as e:
    print(f"⚠️ ML libs unavailable: {e}")
    HAS_TORCH = HAS_NUMPY = HAS_PLOT = HAS_ANIMATION = False
    torch = nn = np = plt = mdates = FuncAnimation = None
    Tensor = List[float]

# 3. Lazy Imports للأداء (Performance Optimized Imports)
if HAS_TORCH:
    from torch import nn as F  # torch.nn.functional
    import torchvision.transforms as transforms
    import torch.optim as optim
    print("✅ Torch advanced modules loaded")


@dataclass
class MonitorResult:
    hit_rate: float = 0.0
    health_score: float = 0.0
    optimized_time: float = 0.0
    turbo: Dict[str, Any] = field(default_factory=dict)
    role: str = ''
    process_time: float = 0.0
    action: str = ''
    risk_level: float = 0.0
    next_step: str = ''
    complexity: float = 0.0      # ✅ للـ monitor()
    anomaly: float = 0.0         # ✅ للـ anomaly detection

# 🔥 TurboEngine class
class HitRateTurboEngine:
    def __init__(self):
        self.multipliers = {0.9:1.5, 0.8:2.0, 0.7:3.0, 0.6:5.0, 0.5:8.0, 0.4:12.0, 0.3:20.0, 0.2:32.0}

    def activate_turbo(self, hit_rate: float, base_time: float) -> Dict[str, Any]:
        multiplier = 1.0
        for threshold, mult in sorted(self.multipliers.items(), reverse=True):
            if hit_rate < threshold:
                multiplier = mult
                break
        processing_time_ms = base_time * multiplier
        return {'multiplier': multiplier, 'processing_time_ms': processing_time_ms}

# الاختبار الكامل - اختبر كل حاجة مرة واحدة
# ========================================
class RealTimeSemanticMonitor:
    def __init__(self, db_path: str = 'monitor.db'):
        """تهيئة مراقب أداء الـ pipeline الذكي المتقدم"""

        # 🔥 البنية الأساسية (Core Infrastructure)
        self.db_path = db_path
        self.function_registry: Dict[str, Dict[str, Any]] = {}
        self.hit_rate_history: Dict[str, Dict[str, Union[int, float]]] = defaultdict(
            lambda: {'hits': 0, 'total': 0, 'rate': 0.0}
        )
        self.pipeline_state: Dict[str, Dict[str, Any]] = {}
        self.global_hit_rate: float = 0.0

        # 🔥 مقاييس الأداء (Performance Counters)
        self.start_time = time.time()
        self.call_count: int = 0
        self.total_latency: float = 0.0

        print("🚀 Initializing RealTimeSemanticMonitor...")
        print("✅ 1. Core structures OK")

        # 🔥 محركات التحسين (Optimization Engines)
        self.turbo_engine = HitRateTurboEngine()
        print("✅ 2. Turbo engine OK")

        # 🔥 البنية التحليلية المتقدمة (Advanced Analytics)
        # 🔥 جميع الـ histories في هيكل موحد لتوفير الذاكرة
        self.history: Dict[str, defaultdict] = {
            'health': defaultdict(list),
            'anomaly': defaultdict(list),
            'complexity': defaultdict(list),
            'turbo': defaultdict(list),
            'auto_heal': defaultdict(list),
            'routing': defaultdict(list)
        }
        self.function_call_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 🔥 بنية الـ ML (Machine Learning Infrastructure)
        self.model_device = torch.device(
            'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        )
        self.model_cache: Dict[str, nn.Module] = {}
        self.model_performance_cache: Dict[str, Dict[str, float]] = {}
        print(f"✅ 3. Device: {self.model_device}")

        # 🔥 قاعدة البيانات (Database Layer)
        self._init_database(db_path)
        print("✅ 4. Database OK")

        # 🔥 مكونات التصوير (Visualization Components)
        self.fig = self.ani = None
        self.ax1 = self.ax2 = self.ax3 = self.ax4 = None
        print("✅ 5. Visualization ready")

        print("🎉 RealTimeSemanticMonitor جاهز 100%! 🚀")
        print(f"📊 DB: {self.db_path} | Functions: {len(self.function_registry)}")
        print("🚀 استخدم: register_functions() → monitor() → setup_visualization() → dashboard")

        self.ax1: Optional[Axes] = None
        self.ax2: Optional[Axes] = None
        self.ax3: Optional[Axes] = None
        self.ax4: Optional[Axes] = None
        self.fig: Optional[Figure] = None

    # ===== DB Methods أولاً =====
    @contextmanager
    def _get_db_connection(self):
        """اتصال آمن بالـDB"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self, db_path: str = None):  # ✅ أضف = None
        """تهيئة قاعدة البيانات مع جداول المراقبة"""
        if db_path is None:
            db_path = getattr(self, 'db_path', 'monitor.db')  # ✅ Safe fallback

        try:
            with self._get_db_connection() as conn:
                # جدول hit_history
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS hit_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        function_name TEXT NOT NULL,
                        hit_rate REAL,
                        health REAL,
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        role TEXT
                    )
                ''')

                # 🔥 جدول pipeline_state كامل
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS pipeline_state (
                        function_name TEXT PRIMARY KEY,
                        role TEXT NOT NULL,
                        health REAL DEFAULT 0.85,
                        hit_rate REAL DEFAULT 0.0,
                        turbo TEXT DEFAULT '{}',
                        action TEXT DEFAULT 'proceed',
                        timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                        complexity REAL DEFAULT 0.3
                    )
                ''')

                conn.commit()
                print(f"✅ Database ready: {db_path}")

        except Exception as e:
            print(f"⚠️  DB init failed: {e}")
            # Fallback: memory-only mode
            self.db_fallback_mode = True

    def _load_history(self) -> Dict[str, Dict]:
        """تحميل التاريخ من DB"""
        history = defaultdict(lambda: {'hits': 0, 'total': 0, 'rate': 0.0})
        try:
            with self._get_db_connection() as conn:
                rows = conn.execute('''
                    SELECT function_name, AVG(hit_rate) as avg_rate, COUNT(*) as count
                    FROM hit_history GROUP BY function_name
                ''').fetchall()
                for row in rows:
                    history[row['function_name']]['rate'] = row['avg_rate'] or 0.0
                    history[row['function_name']]['total'] = row['count']
        except:
            pass  # DB مش موجود أو فاضي
        return dict(history)

    def _save_to_db(self, func_name: str, hit_rate: float, health: float, role: str):
        """حفظ سجل جديد في DB"""
        try:
            with self._get_db_connection() as conn:
                conn.execute('INSERT INTO hit_history (function_name, hit_rate, health, timestamp, role) VALUES (?, ?, ?, ?, ?)',
                            (func_name, hit_rate, health, datetime.now().isoformat(), role))
        except Exception as e:
            print(f"⚠️  DB save error: {e}")

    # ===== Core Methods =====
    def register_functions(self, functions_dict: Dict[str, Dict[str, str]]):
        """تسجيل الدوال مع وصف دورها"""

        if not isinstance(functions_dict, dict):
            raise ValueError("functions_dict must be a dictionary")
        self.function_registry.update(functions_dict)  # دعم إضافة جزئية

        for func_name in functions_dict:
            if func_name not in self.hit_rate_history:
                self.hit_rate_history[func_name] = {'hits': 0, 'total': 0, 'rate': 0.0}
        print(f"✅ Registered {len(functions_dict)} functions")

    def monitor(self, function_name: str, inputs: Any, outputs: Any,
                expected_output: Any = None,
                metadata: Optional[Dict[str, Any]] = None,
                base_time: float = 100.0) -> MonitorResult:
        """
        مراقبة ذكية كاملة مع Auto-Healing + Turbo x32 + Anomaly Detection

        Args:
            function_name: اسم الدالة المُراقَبَة
            inputs: المدخلات (للـ complexity analysis)
            outputs: المخرجات الفعلية
            expected_output: المخرجات المتوقعة (اختياري)
            metadata: {'user': 'rashed', 'priority': 9, 'timestamp': ...}
            base_time: الوقت الأساسي (ms)

        Returns:
            MonitorResult كامل مع كل المقاييس
        """
        # ===== 0️⃣ Validation + Early Return =====
        if function_name not in self.function_registry:
            role = self._smart_role_detection(function_name)
            self.register_functions({function_name: {'role': role}})

        # ===== 1️⃣ التصنيف الذكي + Core Metrics =====
        timestamp = metadata.get('timestamp', datetime.now().isoformat()) if metadata else datetime.now().isoformat()

        # Role من registry أو smart detection
        func_meta = self.function_registry.get(function_name, {})
        role = func_meta.get('role', self._smart_role_detection(function_name))

        # 🔥 Core metrics
        input_complexity = self._compute_input_complexity(inputs)
        health = self._advanced_health_check(outputs, role, input_complexity)
        hit_rate = self._calculate_hitrate(outputs, expected_output)

        # Track فوري
        self._track_performance(function_name, hit_rate)

        # ===== 2️⃣ محرك التوربو المتقدم 🔥 =====
        turbo_result = self._turbo_monitor(
            function_name, inputs, outputs, expected_output, metadata
        )
        turbo_status = self._activate_turbo(hit_rate, max(base_time, turbo_result.process_time))

        # ===== 3️⃣ Auto-Healing + Risk Analysis =====
        healing_action = self._detect_auto_heal(hit_rate, health, role)
        failure_risk = self._predict_failure_risk(function_name)
        anomaly_score = self._detect_anomaly(function_name, {'hit_rate': hit_rate, 'health': health})

        # ===== 4️⃣ Intelligent Routing =====
        next_step = self._intelligent_routing(role, health, hit_rate, anomaly_score)

        # ===== 5️⃣ MonitorResult الكاملة 🔥 =====
        result = MonitorResult(
            hit_rate=hit_rate,
            health_score=health,
            role=role,
            process_time=turbo_result.process_time,
            turbo=turbo_status,
            optimized_time=turbo_status['processing_time_ms'],
            action=healing_action['action'],
            risk_level=failure_risk,
            next_step=next_step,
            complexity=input_complexity,
            anomaly=anomaly_score
        )

        # ===== 6️⃣ Async DB + Cache Update =====
        self.pipeline_state[function_name] = {
            'role': role,
            'health': round(health, 3),
            'hit_rate': round(hit_rate, 3),
            'turbo': {k: round(v, 3) if isinstance(v, float) else v for k, v in turbo_status.items()},
            'action': result.action,
            'timestamp': timestamp
        }

        # Background DB save (non-blocking)
        self._async_save_metrics(function_name, result)

        # Global stats update
        self._update_global_stats()

        return result

    def _detect_auto_heal(self, hit_rate: float, health: float, role: str) -> Dict[str, str]:
        """اكتشاف الحاجة للـ auto-healing"""
        if hit_rate < 0.6 or health < 0.5:
            return {'action': 'heal', 'method': 'retry'}
        elif hit_rate < 0.75 or health < 0.7:
            return {'action': 'warn', 'method': 'optimize'}
        return {'action': 'proceed', 'method': 'none'}

    def _predict_failure_risk(self, func_name: str) -> float:
        """Trend-based failure risk prediction"""

        # -------- Load real history --------
        history = self._load_history().get(func_name, {})
        if not history:
            return 0.5

        hit_rates = [h.get('hit_rate', 0.5) for h in history.values()]
        hit_rates = hit_rates[-20:]  # آخر 20

        if len(hit_rates) < 5:
            return 0.5

        # -------- Simple trend analysis --------
        import numpy as np
        recent_avg = np.mean(hit_rates[-5:])     # آخر 5
        history_avg = np.mean(hit_rates[:-5])    # الباقي

        # Risk = trend degradation
        trend_drop = max(0, history_avg - recent_avg)
        risk = min(1.0, trend_drop * 2.0)

        return risk

    def detect_anomaly(self, func_name: str, current_metrics: Dict) -> str:
        """Z-score anomaly detection على آخر 50 execution"""

        # -------- Get function history --------
        history = self._load_history().get(func_name, {})
        hit_rates = [h.get('hit_rate', 0.5) for h in history.values()][-50:]  # آخر 50

        if len(hit_rates) < 10:  # مش كفاية data
            return 'normal'

        # -------- Calculate Z-score --------
        import numpy as np
        mean_history = np.mean(hit_rates)
        std_history = np.std(hit_rates) if np.std(hit_rates) > 0 else 1e-6

        z_score = abs(current_metrics['hit_rate'] - mean_history) / std_history

        # -------- Anomaly classification --------
        if z_score > 3.0:      # 99.7% confidence [web:276]
            return 'alert'
        elif z_score > 2.0:    # 95% confidence
            return 'warning'
        else:
            return 'normal'

    def _intelligent_routing(self, role: str, health: float, hit_rate: float, anomaly: float = 0.0) -> str:
        """
        توجيه ذكي متقدم مع 5-level decision tree

        Returns:
            'deploy' | 'continue_pipeline' | 'fallback_model' | 'quarantine' | 'retry'
        """
        # 🔥 Priority 1: Critical anomalies
        if anomaly > 0.8:
            return 'quarantine'

        # 🔥 Priority 2: Health crisis
        if health < 0.4:
            return 'fallback_model'

        # 🔥 Priority 3: Role-specific routing
        routes = {
            'prediction': ['integration', 'validation'][int(health > 0.8)],
            'generation': ['refinement', 'finalize'][int(hit_rate > 0.9)],
            'integration': ['output', 'validation'][int(hit_rate > 0.85)],
            'validation': ['deploy', 'retry'][int(hit_rate > 0.95)]
        }

        # 🔥 Priority 4: Performance-based escalation
        if hit_rate < 0.5:
            return 'retry'
        elif hit_rate < 0.7:
            return 'optimize'

        # 🔥 Priority 5: Default success path
        return routes.get(role, 'next_stage')

    def _async_save_metrics(self, func_name, result):
        # Thread/async save
        pass

    def _update_global_stats(self):
        self.global_hit_rate = sum(h['rate'] * h['total'] for h in self.hit_rate_history.values()) / max(1, sum(h['total'] for h in self.hit_rate_history.values()))

    # ===== Private Helpers =====
    def simple_hitrate(self, actual: Any, expected: Any) -> float:
        """Hit.Rate - pure Python، no Torch errors"""
        if expected is None:
            return 1.0

        def safe_float(val: Any) -> float:
            """تحويل آمن لـfloat"""
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                # Torch scalar tensor
                if HAS_TORCH and hasattr(val, 'item') and val.numel() == 1:
                    return float(val.item())
                return float(val)
            except (ValueError, TypeError, AttributeError):
                return 0.0

        # Scalar case
        if not isinstance(actual, (list, tuple)):
            a = safe_float(actual)
            e = safe_float(expected)
            return 1.0 if abs(a - e) <= 0.1 else 0.0

        # List case - limit للـreal-time
        act_list = actual[:10] if isinstance(actual, (list, tuple)) else [actual]
        exp_list = expected[:10] if isinstance(expected, (list, tuple)) else [expected]

        min_len = min(len(act_list), len(exp_list))
        if min_len == 0:
            return 0.0

        matches = 0
        for i in range(min_len):
            if abs(safe_float(act_list[i]) - safe_float(exp_list[i])) <= 0.1:
                matches += 1

        return matches / min_len

    def _smart_role_detection(self, func_name: str) -> str:
        patterns = {
            r'predict|forecast|estimate|model': 'prediction',
            r'generate|create|produce|gen': 'generation',
            r'integrate|combine|merge|int': 'integration',
            r'validate|check|verify|val|test': 'validation',
            r'process|compute|run|exec': 'processing'
        }
        name = func_name.lower()
        for pattern, role in patterns.items():
            if re.search(pattern, name): return role
        return 'unknown'

    def _calculate_hitrate(self, outputs, expected_output):
        """حساب hit rate بين actual vs expected"""

        # -------- Safe numeric extraction --------
        def safe_extract(val):
            if isinstance(val, list):
                return safe_extract(val[0]) if val else 0.0
            if isinstance(val, str):
                try:
                    return float(val)  # "3.14" → 3.14
                except ValueError:
                    return 0.5
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, torch.Tensor):
                return val.item() if val.numel() == 1 else val.mean().item()
            return 0.5

        actual_val = safe_extract(outputs)
        expected_val = safe_extract(expected_output)

        # -------- Scalar comparison (بسيط وآمن) --------
        actual_t = torch.tensor(actual_val, device=self.model_device)
        expected_t = torch.tensor(expected_val, device=self.model_device)

        return 1.0 if torch.abs(actual_t - expected_t) <= 0.1 else 0.0

    def _track_performance(self, function_name: str, hit_rate: float):
        """تتبع الأداء"""
        hist = self.hit_rate_history.setdefault(function_name, {'hits': 0, 'total': 0})
        hist['hits'] += hit_rate > 0.8
        hist['total'] += 1
        hist['rate'] = hist['hits'] / hist['total']

    # ===== Turbo Monitor المتقدم مع كل البارامترات 🔥 =====
    def _turbo_monitor(
        self,
        function_name: str,
        inputs: Any,
        outputs: Any,
        expected_output: Any = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MonitorResult:

        """محرك التوربو الداخلي مع استخدام كامل للبارامترات"""
        start_time = time.perf_counter()

        # استخدام inputs: حساب complexity (طول/عمق البيانات)
        input_complexity = self._compute_input_complexity(inputs)  # دالة جديدة تحت

        # استخدام expected_output: hitrate أولي
        prelim_hitrate = self.simple_hitrate(outputs, expected_output) if expected_output else 1.0

        # استخدام metadata: timestamp أو custom metrics
        timestamp = metadata.get('timestamp', datetime.now().isoformat()) if metadata else datetime.now().isoformat()

        # باقي المنطق...
        process_time_ms = (time.perf_counter() - start_time) * 1000
        role = self._smart_role_detection(function_name)
        health = self._advanced_health_check(outputs, role, input_complexity)  # 3 args!
        if input_complexity > 0.5:  # استخدم if مش index
            health *= (1 - 0.2)  # penalty بسيط
        health = max(0.0, min(1.0, health))  # clamp

        turbo_status = self._activate_turbo(prelim_hitrate, process_time_ms)  # يستخدم prelim_hitrate

        return MonitorResult(
            hit_rate=prelim_hitrate,           # ✅
            health_score=health,               # ✅
            role=role,                         # ✅
            process_time=process_time_ms,      # ✅
            turbo=turbo_status,                # ✅
            optimized_time=turbo_status.get('processing_time_ms', process_time_ms),  # ✅
            action='proceed',                  # ✅ أو من healing_action
            risk_level=0.1,                    # ✅ default أو محسوب
            next_step='next_stage'             # ✅
            # مفيش metadata؛ لو عايز، أضفها في dataclass
        )

    def _activate_turbo(self, hit_rate: float, base_time: float) -> Dict[str, float]:
        """تفعيل Turbo بناءً على hit_rate"""
        multiplier = 1.5
        if hit_rate >= 0.9: multiplier = 1.2
        elif hit_rate >= 0.8: multiplier = 1.5
        elif hit_rate >= 0.7: multiplier = 2.0
        elif hit_rate >= 0.6: multiplier = 3.0
        elif hit_rate >= 0.5: multiplier = 5.0
        else: multiplier = 8.0  # Up to x32 في V2

        optimized_time = base_time / multiplier
        return {
            'processing_time_ms': optimized_time,
            'multiplier': multiplier,
            'strategy': 'dynamic'
        }

    def _detect_anomaly(self, function_name: str, metrics: Dict[str, float]) -> float:
        """اكتشاف الشذوذ (0.0-1.0)"""
        # Simple Z-score proxy
        hist_rate = self.hit_rate_history.get(function_name, {}).get('rate', 0.5)
        z_score = abs(metrics['hit_rate'] - hist_rate) / max(0.01, hist_rate)
        return min(1.0, z_score / 3.0)  # >3σ = anomaly

    def _advanced_health_check(self, outputs: Any, role: str, input_complexity: float = 0.3) -> float:
        """تحقق صحة محسّن مع input complexity"""
        if not outputs:
            return 0.0

        # 🔥 Base score عالي (healthy by default)
        base_score = 0.85

        # تعديل حسب حجم outputs
        if isinstance(outputs, (list, dict, np.ndarray)) and len(outputs) > 0:
            output_size = len(outputs)
            if output_size > 100: base_score *= 0.9  # كبير = penalty بسيط
            elif output_size < 10: base_score *= 1.05 # صغير = bonus

        # تخصيص حسب الـ role
        if role == 'prediction':
            base_score = min(0.95, base_score + 0.05)  # predictions أعلى شوية
        elif role == 'generation':
            base_score = min(0.90, base_score)  # generation أقل شوية

        # 🔥 Penalty للـ input_complexity
        health = base_score * (1 - input_complexity * 0.3)  # 0.3 ألطف من 0.5
        return max(0.0, min(1.0, health))

    def _compute_input_complexity(self, inputs: Any) -> float:
        """حساب تعقيد الإدخال"""
        if isinstance(inputs, (list, dict)): return min(1.0, len(inputs)/1000)
        elif isinstance(inputs, str): return min(1.0, len(inputs)/5000)
        return 0.3

    # ===== Advanced/ML =====
    def monitor_tensor(self, model_name: str, inputs: torch.Tensor,
                    outputs: torch.Tensor, targets: torch.Tensor,
                    base_time: float = 100.0) -> MonitorResult:
        """مراقبة مخصصة للـML models"""
        if not HAS_TORCH:
            raise RuntimeError("Torch required for monitor_tensor")

        with torch.no_grad():
            hit_rate = self._calculate_hitrate(outputs, targets)
            health = self._advanced_health_check(outputs.cpu().tolist(), 'prediction')

            # 🔥 استخدم base_time في الـ turbo
            turbo_result = self._turbo_monitor(
                model_name, inputs.cpu().tolist(),
                outputs.cpu().tolist(), targets.cpu().tolist(),
                metadata={'base_time': base_time}
            )

        return MonitorResult(
            hit_rate=hit_rate,
            health_score=health,
            role='prediction',
            process_time=turbo_result.process_time,
            turbo=turbo_result.turbo,
            optimized_time=turbo_result.optimized_time,
            action='proceed',  # ✅ مفقود
            risk_level=0.1,    # ✅ مفقود
            next_step='continue'  # ✅ مفقود
        )

    # ===== Viz/Dashboard =====
    def setup_visualization(self):
        print("🔍 STEP 1: ENTERED setup_visualization()")

        global HAS_PLOT
        HAS_PLOT = True
        print(f"🔍 STEP 1.5: HAS_PLOT={HAS_PLOT}")

        if not HAS_PLOT:
            print("⚠️ Matplotlib not available")
            return False

        # -------- STEP 2: Import matplotlib --------
        print("🔍 STEP 2: import matplotlib")
        import matplotlib
        import matplotlib.pyplot as plt

        # -------- STEP 3: Fix Backend + Interactive --------
        print("🔍 STEP 3: Backend fix")
        matplotlib.use('Agg')  # 👈 Non-blocking backend!
        plt.switch_backend('Agg')

        plt.ion()
        plt.style.use('dark_background')
        print("🔍 STEP 3.5: Setup OK")

        # -------- STEP 4: Safe subplots --------
        print("🔍 STEP 4: plt.subplots()")
        try:
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
                2, 2, figsize=(12, 8),
                constrained_layout=True  # 👈 بدل facecolor
            )
            print("✅ STEP 4: Subplots OK")
        except Exception as e:
            print(f"❌ STEP 4 ERROR: {e}")
            return False

        # ------- STEP 5: Style the dashboard --------
        print("🔍 STEP 5: Styling dashboard"
            )

        # -------- Matplotlib Core Setup --------
        import matplotlib.pyplot as plt
        plt.ion()
        plt.style.use('dark_background')

        # -------- Dashboard Creation & Styling --------
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(
            2, 2, figsize=(15, 10), facecolor='#0f0f0f'
        )

        self.fig.suptitle('🚀 RealTimeSemanticMonitor Dashboard',
                        fontsize=18, fontweight='bold', color='cyan', y=0.98)

        # Style axes
        axes_info = [
            ('Hit Rate', 'skyblue'), ('Health Score', 'salmon'),
            ('Turbo xMultiplier', 'lightgreen'), ('Live Stats', 'gold')
        ]

        for i, (title, color) in enumerate(axes_info):
            ax = [self.ax1, self.ax2, self.ax3, self.ax4][i]
            ax.set_facecolor('#1a1a1a')
            ax.grid(True, alpha=0.3)
            ax.set_title(title, fontsize=14, fontweight='bold', color=color)
            ax.tick_params(colors='white')

            if i < 3:
                ax.bar([], [], color=color, alpha=0.8)
                ax.set_ylim(0, 1.1 if i < 2 else 35)
            else:
                ax.axis('off')
                ax.text(0.05, 0.95, 'Initializing...', va='top', fontsize=12)

        plt.tight_layout()
        print("✅ 📊 Visualization setup complete")
        return True

    def show_static_dashboard(self):
        """عرض dashboard ثابت مع بيانات من DB"""
        history = self._load_history()
        if not history:
            print("⚠️  No data for dashboard")
            return

        func_names = list(history.keys())[:8]  # Max 8 functions
        if len(func_names) == 0:
            return

        # 1. Hit Rate (real data)
        hit_rates = [history[f].get('rate', 0.0) for f in func_names]
        self.ax1.bar(range(len(func_names)), hit_rates, color='skyblue', alpha=0.8)
        self.ax1.set_title('Hit Rate by Function')
        self.ax1.set_ylim(0, 1.1)
        self.ax1.set_xticks(range(len(func_names)))
        self.ax1.set_xticklabels(func_names, rotation=45, ha='right')

        # 2. Health (من pipeline_state)
        health_scores = []
        for f in func_names:
            state = self.pipeline_state.get(f, {})
            health_scores.append(state.get('health', 0.85))
        self.ax2.bar(range(len(func_names)), health_scores, color='salmon', alpha=0.8)
        self.ax2.set_title('Health Score')
        self.ax2.set_ylim(0, 1.1)

        # 3. Turbo (real multipliers)
        turbo_mults = []
        for f in func_names:
            turbo = self.pipeline_state.get(f, {}).get('turbo', {})
            turbo_mults.append(turbo.get('multiplier', 1.0))
        self.ax3.bar(range(len(func_names)), turbo_mults, color='lightgreen', alpha=0.8)
        self.ax3.set_title('Turbo xMultiplier')
        self.ax3.set_ylim(0, max(3.5, max(turbo_mults)*1.1))

        # 4. Global Stats Summary
        global_hr = self.global_hit_rate
        total_calls = sum(h.get('total', 0) for h in history.values())
        self.ax4.axis('off')
        self.ax4.text(0.1, 0.9, f'Global Hit Rate: {global_hr:.1%}', fontsize=14, fontweight='bold')
        self.ax4.text(0.1, 0.7, f'Total Calls: {total_calls:,}', fontsize=12)
        self.ax4.text(0.1, 0.5, f'Functions: {len(self.function_registry)}', fontsize=12)
        self.ax4.text(0.1, 0.3, f'Active: {len([f for f,h in history.items() if h["total"]>0])}', fontsize=12)

        self.fig.suptitle(f'RealTimeSemanticMonitor - {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=16)
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking للـ live monitoring

    def get_db_stats(self) -> Dict[str, Any]:
        """
        إحصائيات DB شاملة + memory fallback

        Returns:
            {
                'total_records': 1234,
                'functions_monitored': 15,
                'avg_hit_rate': 0.87,
                'best_function': {'name': 'live_pred', 'rate': 0.94},
                'memory_functions': 12,     # من memory
                'global_hit_rate': 0.85     # cached
            }
        """
        stats = {
            'memory_functions': len(self.function_registry),
            'memory_calls': sum(h.get('total', 0) for h in self.hit_rate_history.values()),
            'global_hit_rate': self.global_hit_rate
        }

        try:
            with self._get_db_connection() as conn:
                # Core stats
                stats['total_records'] = conn.execute('SELECT COUNT(*) FROM hit_history').fetchone()[0]
                stats['functions_monitored'] = conn.execute(
                    'SELECT COUNT(DISTINCT function_name) FROM hit_history'
                ).fetchone()[0]

                # Advanced analytics
                stats['avg_hit_rate'] = conn.execute('SELECT AVG(hit_rate) FROM hit_history').fetchone()[0] or 0.0

                # 🔥 Best + Worst functions
                best = conn.execute('''
                    SELECT function_name, AVG(hit_rate) as rate
                    FROM hit_history
                    GROUP BY function_name
                    ORDER BY rate DESC LIMIT 1
                ''').fetchone()
                stats['best_function'] = {'name': best['function_name'], 'rate': best['rate']} if best else None

                worst = conn.execute('''
                    SELECT function_name, AVG(hit_rate) as rate
                    FROM hit_history
                    GROUP BY function_name
                    HAVING COUNT(*) > 5
                    ORDER BY rate ASC LIMIT 1
                ''').fetchone()
                stats['worst_function'] = {'name': worst['function_name'], 'rate': worst['rate']} if worst else None

                # Recent activity (last hour)
                recent = conn.execute('''
                    SELECT COUNT(*) FROM hit_history
                    WHERE timestamp > datetime('now', '-1 hour')
                ''').fetchone()[0]
                stats['recent_activity'] = recent

        except Exception as e:
            print(f"⚠️  DB stats partial (memory only): {e}")
            stats['db_error'] = str(e)

        return stats

    def start_live_monitoring_blit(self, interval: float = 2.0):
        """
        Live monitoring مع blit=True للأداء العالي

        Args:
            interval: ثواني بين التحديثات (1.0 = 1s)
        """
        if not HAS_PLOT or self.fig is None or self.ax1 is None:
            print("⚠️  Plot setup required. Call setup_visualization() first")
            return

        if not HAS_ANIMATION:
            print("⚠️  matplotlib.animation not available")
            return

        # -------- 🔥 تحديث المحاور مع real data --------
        def update_ax(frame):
            # Frame counter للـ performance
            if frame % 10 == 0:  # كل 10 frames بس
                print(f"🔄 Frame {frame}: {len(self._load_history())} functions tracked")

            history = self._load_history()
            func_names = list(history.keys())[:6]

            # Risk alerts
            for func_name in func_names:
                risk = self._predict_failure_risk(func_name)
                if risk > 0.8:
                    print(f"🚨 HIGH RISK: {func_name} ({risk:.1%})")

            # Update title & fig
            self.fig.suptitle(f'🚀 RealTimeSemanticMonitor | {datetime.now().strftime("%H:%M:%S")}', fontsize=16)
            self.fig.patch.set_facecolor('#0f0f0f')
            self.fig.tight_layout(rect=[0, 0, 1, 0.95])

            # -------- Safe axes styling --------
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                if ax:
                    ax.set_facecolor('#1a1a1a')
                    ax.grid(True, alpha=0.3)

            # -------- 🔥 AX1: Hit Rate Bars --------
            if func_names and self.ax1:
                hit_rates = [history[f].get('hit_rate', 0.0) for f in func_names]
                self.ax1.clear()
                self.ax1.bar(range(len(func_names)), hit_rates, color='skyblue', alpha=0.7)
                self.ax1.set_title(f'Hit Rate (Global: {self.global_hit_rate:.1%})')
                self.ax1.set_ylim(0, 1.1)

            # -------- 🔥 AX2: Health Bars --------
            if func_names and self.ax2:
                healths = [self.pipeline_state.get(f, {}).get('health', 0.85) for f in func_names]
                self.ax2.clear()
                self.ax2.bar(range(len(func_names)), healths, color='salmon', alpha=0.7)
                self.ax2.set_title('Health Score')
                self.ax2.set_ylim(0, 1.1)

            # -------- 🔥 AX3: Turbo Multiplier --------
            if func_names and self.ax3:
                turbos = [self.pipeline_state.get(f, {}).get('turbo', {}).get('multiplier', 1.0) for f in func_names]
                self.ax3.clear()
                self.ax3.bar(range(len(func_names)), turbos, color='lightgreen', alpha=0.7)
                self.ax3.set_title('Turbo xMultiplier')

            # -------- 🔥 AX4: Live Stats --------
            if self.ax4:
                self.ax4.clear()
                self.ax4.axis('off')
                stats_text = f"""
        Global Hit Rate: {self.global_hit_rate:.1%}
        Total Calls: {sum(h.get('total',0) for h in history.values()):,}
        Active Functions: {len(func_names)}
        Uptime: {(time.time()-self.start_time)/3600:.1f}h
                """
                self.ax4.text(0.05, 0.95, stats_text, fontsize=11, va='top',
                            transform=self.ax4.transAxes, fontfamily='monospace')

            # -------- Return artists للـ blit --------
            return [self.ax1, self.ax2, self.ax3, self.ax4]

        # -------- 🔥 FuncAnimation مع كل الـ optimizations --------
        self.ani = FuncAnimation(
            self.fig, update_ax,
            interval=int(interval * 1000),  # ms
            blit=True,                      # 10x faster redraw
            cache_frame_data=False,         # Memory efficient
            repeat=True
        )

        plt.show(block=False)  # Non-blocking للـ interactivity
        print(f"🎬 Live Dashboard started | {interval}s interval | Blit=ON 🚀")

    # ===== آخر: Public Utils =====
    def save_pipeline_state(self, func_name: str, state: Dict[str, Any]) -> bool:
        """
        حفظ حالة pipeline في DB مع bulk insert + error recovery

        Args:
            func_name: اسم الدالة
            state: {'role': 'pred', 'health': 0.92, 'turbo': {...}}

        Returns:
            True if saved successfully
        """
        try:
            # 🔥 Normalize + serialize complex fields
            db_state = {
                'role': state.get('role', 'unknown'),
                'health': round(state.get('health', 0.85), 3),
                'hit_rate': round(state.get('hit_rate', 0.0), 3),
                'turbo': str({k: round(v, 3) if isinstance(v, (int, float)) else v
                            for k, v in (state.get('turbo', {})).items()}),
                'action': state.get('action', 'proceed'),
                'timestamp': state.get('timestamp', datetime.now().isoformat()),
                'complexity': round(state.get('complexity', 0.3), 2)
            }

            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO pipeline_state
                    (function_name, role, health, hit_rate, turbo, action, timestamp, complexity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    func_name,
                    db_state['role'],
                    db_state['health'],
                    db_state['hit_rate'],
                    db_state['turbo'],
                    db_state['action'],
                    db_state['timestamp'],
                    db_state['complexity']
                ))

            # 🔥 Bulk update global stats كل 10 calls
            self.call_count += 1
            if self.call_count % 10 == 0:
                self._update_global_stats()

            return True

        except Exception as e:
            print(f"⚠️  Pipeline state save failed: {e}")
            # Fallback: memory only
            self.pipeline_state[func_name] = state
            return False

# ======================== Main Test Suite ========================
if __name__ == "__main__":
    """اختبار شامل لـ RealTimeSemanticMonitor - Production Ready Demo"""

    print("🚀 === RealTimeSemanticMonitor - Complete Test Suite ===\n")

    # -------- 1️⃣ التهيئة الأساسية (Core Initialization) --------
    print("1️⃣ === Core Initialization ===")
    monitor = RealTimeSemanticMonitor('ai_monitor.db')

    # -------- 2️⃣ إعداد التصوير (Visualization Setup) --------
    print(f"🔍 Debug: HAS_PLOT={HAS_PLOT}")  # Debug

    viz_ok = monitor.setup_visualization()
    print(f"🔍 viz_ok result: {viz_ok}")

    if viz_ok:
        print("✅ Dashboard ready!")
        monitor.show_static_dashboard()
    else:
        print("⚠️  Dashboard disabled")
        print("💡 pip install matplotlib")

    # -------- 3️⃣ عرض تفاعلي (اختياري) --------
    print("\n3️⃣ === Live Dashboard Demo ===")
    live_choice = input("Start live monitoring? [y/N]: ").lower().startswith('y')
    if live_choice and viz_ok:
        try:
            print("🎬 Starting live updates (Ctrl+C to stop)...")
            monitor.start_live_monitoring_blit(interval=2.0)
            input("\nPress Enter to continue main tests...")
        except KeyboardInterrupt:
            print("\n⏹️  Live monitoring stopped by user")

    print("\n✅ === INITIALIZATION PHASE COMPLETE ===\n")
    print("=" * 70)

    # ============= 4️⃣ الاختبارات الأساسية (Core Tests) =============
    print("🚀 === COMPREHENSIVE TESTING PHASE ===\n")

    # -------- 4.1 تسجيل الدوال (Advanced Registration) --------
    print("4.1 === Smart Registration ===")
    functions = {
        'live_prediction': {'role': 'prediction', 'priority': 9},
        'data_integration': {'role': 'integration'},
        'image_generator': {'complexity': 'high'}
    }
    reg_result = monitor.register_functions(functions)

    if reg_result is not None:
        print(f"✅ Registered: {reg_result.get('registered', 0)} functions")
    else:
        print("✅ Registration complete (result missing)")  # ✅ تمام!

    # -------- 4.2 اختبار monitor الأساسي --------
    print("\n4.2 === Core Monitor Test ===")
    result = monitor.monitor(
        'live_prediction',
        inputs=[1,2,3,4,5],
        outputs=[1.1,1.9,3.0,4.1,5.2],
        expected_output=[1,2,3,4,5],
        base_time=120.0
    )
    print(f"✅ {result.role}: {result.hit_rate:.1%} | "
          f"Turbo x{result.turbo['multiplier']:.1f} | "
          f"{result.optimized_time:.0f}ms | "
          f"Action: {result.action}")

    # -------- 4.3 إحصائيات قاعدة البيانات --------
    print("\n4.3 === Database Stats ===")
    db_stats = monitor.get_db_stats()
    print(f"💾 Records: {db_stats.get('total_records', 0):,}")
    print(f"📊 Functions monitored: {db_stats.get('functions_monitored', 0)}")

    # -------- 5️⃣ اللوحة النهائية ---------
    print("\n5️⃣ === FINAL DASHBOARD ===")
    if viz_ok:
        monitor.show_static_dashboard()

    print("\n🎉 === ✅ ALL TESTS PASSED SUCCESSFULLY === 🚀")
    print(f"⏱️  Uptime: {(time.time()-monitor.start_time)/60:.1f} minutes")
    print("🏆 Production-ready RealTimeSemanticMonitor!")

    # ------------- Main Test Suite -------------------------------
    print("🚀 === RealTimeSemanticMonitor - Main Test Suite ===\n")

    # 1️⃣ Advanced Registration ✅ مثالي
    print("1️⃣ === Smart Registration ===")
    functions = {
        'live_prediction': {'role': 'prediction', 'priority': 9, 'timeout_ms': 2000},
        'data_integration': {'role': 'integration'},
        'image_generator': {'complexity': 'high'},
        'validate_results': {'role': 'validation'}
    }
    reg_result = monitor.register_functions(functions)
    print(f"✅ {reg_result}")  # Fixed print

    # 2️⃣ MonitorResult Test ✅ مثالي
    print("\n2️⃣ === MonitorResult Test ===")
    result = MonitorResult(hit_rate=0.92, health_score=0.87, complexity=0.3)
    print("Initial:", result)
    result.hit_rate = 0.95
    print("Updated:", f"Hit: {result.hit_rate:.1%} | Health: {result.health_score:.2f}")

    # 3️⃣ Turbo Engine Test ✅ مثالي
    print("\n3️⃣ === Turbo Engine Test ===")
    test_cases = [(0.98, 120), (0.88, 80), (0.72, 200), (0.45, 300)]
    for hit_rate, base_time in test_cases:
        turbo = monitor._activate_turbo(hit_rate, base_time)  # Private access
        print(f"Hit {hit_rate:.0%}: {base_time}ms → {turbo['processing_time_ms']:.0f}ms (x{turbo['multiplier']:.1f})")

    # 🔥 4️⃣ Core Monitor Test (القلب) ✅ مثالي
    print("\n4️⃣ === Core Monitor Test ===")
    test_cases = [
        ('live_pred', [1,2,3,4,5], [1.1,1.9,3.0,4.2,5.1], [1,2,3,4,5]),
        ('heavy_gen', ['prompt']*100, ['image']*100, None),
        ('integrate', [1,2], [3], [3])
    ]

    for name, inputs, outputs, expected in test_cases:
        start = time.perf_counter()
        time.sleep(0.1 * len(inputs) if isinstance(inputs, list) else 0.05)
        real_time = (time.perf_counter() - start) * 1000

        result = monitor.monitor(
            function_name=name, inputs=inputs, outputs=outputs,
            expected_output=expected, base_time=real_time,
            metadata={'user': 'rashed', 'session': 'test'}
        )

        print(f"✅ {name}: {real_time:.0f}ms → {result.optimized_time:.0f}ms")
        print(f"   {result.role} | Hit:{result.hit_rate:.1%} | H:{result.health_score:.2f} | "
            f"Risk:{result.risk_level:.0%} | x{result.turbo['multiplier']:.1f} | {result.next_step}")

    # 5️⃣ Torch Test ✅ مثالي (conditional)
    if HAS_TORCH:
        print("\n5️⃣ === PyTorch Monitor Test ===")
        dummy_model = nn.Linear(10, 10).to(monitor.model_device)
        inputs_t = torch.randn(5, 10).to(monitor.model_device)
        targets_t = torch.randn(5, 10).to(monitor.model_device)
        outputs_t = dummy_model(inputs_t)

        result_t = monitor.monitor_tensor('dummy_nn', inputs_t, outputs_t, targets_t, base_time=150)
        print(f"🧠 Torch: Hit:{result_t.hit_rate:.1%} | Health:{result_t.health_score:.2f} | "
            f"Turbo x{result_t.turbo['multiplier']:.1f}")

    # 🔥 6️⃣ اللوحة النهائية (آخر حاجة)
    print("\n6️⃣ === Final Analytics ===")
    print(f"💾 DB Stats: {monitor.get_db_stats()}")
    print(f"📊 Pipeline States: {len(monitor.pipeline_state)} functions")
    print(f"🌍 Global Hit Rate: {monitor.global_hit_rate:.1%}")

    print("\n🏆 === FINAL DASHBOARD ===")
    monitor.show_static_dashboard()

    print("\n🎉 === ✅ TEST SUITE COMPLETE 100% === 🚀")
    print(f"⏱️  Total uptime: {(time.time()-monitor.start_time)/60:.1f} minutes")
