
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json, os, re, time, unicodedata, threading, webbrowser
import numpy as np
import httpx
from collections import deque, Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ── Dependencias Opcionales ──
try:
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import joblib
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    _CC_AVAILABLE = True
except ImportError:
    _CC_AVAILABLE = False

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    _API_AVAILABLE = True
except ImportError:
    _API_AVAILABLE = False

VERSION = "8.1-PlanAligned"

# ══════════════════════════════════════════════════════════════════════════════
#  HIPERPARÁMETROS (Sensibilidad de la Tesis)
# ══════════════════════════════════════════════════════════════════════════════
ALPHA_DEFAULT    = 0.30   # Tasa de suavizado del promedio móvil exponencial (EWMA)
COUPLING_DEFAULT = 0.50   # Fuerza de acoplamiento matricial c
THETA_QUIEBRE    = 0.50   # Umbral de detección de quiebre temático (Similitud Coseno)
THETA_BAJO       = 0.66   # Entropía < 0.66 -> directiva directa (colapso forzado)
THETA_ALTO       = 0.81   # Entropía > 0.81 -> directiva exploratoria (superposición)
N_DIMS           = 12

FACTOR_EMISION_EC = 0.1616 # kgCO2eq/kWh Ecuador


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSIONES SEMÁNTICAS (H^12)
# ══════════════════════════════════════════════════════════════════════════════
DIM_SEMANTICS: Dict[int, Tuple[str, str, List[str]]] = {
    0:  ("emocion", "Emocionalidad / bienestar",
         ["feliz","triste","cansado","estres","estresado","bien","mal","amor","miedo",
          "alegre","ansioso","ansiedad","frustrado","contento","preocupado","agotado",
          "motivado","dolor","molesto","enojado","tranquilo","nervioso","siento",
          "me siento","angustia","abrumado"]),
    1:  ("logica", "Lógica / código / tecnología",
         ["codigo","python","funcion","error","bug","api","clase","variable","algoritmo",
          "datos","database","servidor","sql","loop","for","while","select","query",
          "join","tabla","javascript","react","git","docker","numpy","pandas","script",
          "libreria","framework","pip","json","endpoint","compilar","debug"]),
    2:  ("incertidumbre", "Dudas / preguntas / exploración",
         ["como","que","porque","cuando","donde","cual","podria","deberia","quizas",
          "no se","duda","pregunta","explicar","entender","diferencia","posible",
          "no entiendo","confundido","como funciona","que es","para que sirve"]),
    3:  ("tiempo", "Urgencia / planificación / plazos",
         ["manana","hoy","despues","antes","urgente","rapido","plazo","plan","futuro",
          "ahora","luego","lento","tarde","deadline","semana","mes","critico",
          "bloqueado","lo antes posible","cuanto tiempo"]),
    4:  ("creacion", "Creatividad / diseño / construcción",
         ["crear","disenar","inventar","nuevo","idea","innovar","construir","implementar",
          "generar","desarrollar","prototipar","refactorizar","agregar","integrar",
          "combinar","hacer","armar"]),
    5:  ("analisis", "Análisis / comparación / evaluación",
         ["comparar","analizar","evaluar","diferencia","mejor","peor","ventaja",
          "desventaja","resultado","conclusion","versus","contra","pros","contras",
          "benchmark","rendimiento","performance","vs","distingue"]),
    6:  ("conexion", "Continuidad / contexto / referencias previas",
         ["entonces","ademas","sin embargo","aunque","relacion","contexto","anterior",
          "seguir","continuar","como decia","retomando","mencionaste","recuerdas",
          "sobre lo mismo","a proposito","volviendo"]),
    7:  ("profundidad", "Detalle técnico / especificidad / precisión",
         ["especifico","detalle","exactamente","tecnico","avanzado","profundo","complejo",
          "arquitectura","estructura","memoria","precision","matematico","definicion",
          "implementacion","bajo nivel","fundamento","a fondo"]),
    8:  ("social", "Colaboración / interacción humana",
         ["nosotros","equipo","juntos","ayuda","colaborar","compartir","usuario","cliente",
          "persona","comunidad","empresa","colega","proyecto","jefe","companero","reunion"]),
    9:  ("aprendizaje", "Estudio / comprensión / crecimiento",
         ["aprender","estudiar","entender","saber","conocimiento","curso","tutorial",
          "ejemplo","practica","ejercicio","explicame","recomendar","recomendacion",
          "guia","recurso","libro","examen","clase","ensename"]),
    10: ("decision", "Elección / recomendación / dilema",
         ["elegir","decidir","opcion","alternativa","escoger","consejo","sugiere",
          "prefiero","cual es mejor","me recomiendas","dilema","deberia usar",
          "cual usar","recomendarias","conviene"]),
    11: ("meta", "Objetivos / propósito / resultados concretos",
         ["objetivo","meta","lograr","conseguir","resultado","quiero","necesito","busco",
          "dame","muestrame","genera","quiero que","hazme","escribe","implementa",
          "desarrolla","construye","pon"]),
}

# Matriz de acoplamiento semántico 12x12
COUPLING_LINKS = [
    (0, 8, 0.65),   # emocion    <-> social
    (0, 9, 0.35),   # emocion    <-> aprendizaje
    (1, 5, 0.70),   # logica     <-> analisis
    (1, 7, 0.65),   # logica     <-> profundidad
    (1, 4, 0.50),   # logica     <-> creacion
    (2, 9, 0.60),   # incertid.  <-> aprendizaje
    (2, 10, 0.55),  # incertid.  <-> decision
    (3, 11, 0.70),  # tiempo     <-> meta
    (5, 10, 0.75),  # analisis   <-> decision
    (6, 8, 0.50),   # conexion   <-> social
]

def _build_coupling() -> np.ndarray:
    C = np.eye(N_DIMS)
    for i, j, w in COUPLING_LINKS:
        C[i, j] = C[j, i] = w
    return C

COUPLING_MATRIX = _build_coupling()

INTENT_TO_DIMS = {
    "tecnologia":     [1, 5, 7],
    "trabajo":        [3, 8, 11],
    "estudio":        [2, 9, 10],
    "bienestar":      [0, 6, 8],
    "conversacional": [6, 8, 2],
    "general":        [2, 4, 6],
}
ML_PROJECTION_WEIGHT = 0.40

# ══════════════════════════════════════════════════════════════════════════════
#  SEÑAL DE CONTEXTO (ESTADO DEL SISTEMA COMPATIBLE CON DASHBOARD v6.1)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ContextSignal:
    turn_number: int = 0
    exec_time_ms: float = 0.0
    mode: str = "rapido"

    # Métricas principales para Dashboard
    entropy_vn: float = 0.0            # Entropía Shannon
    fidelity: float = 1.0              # Fidelidad Semántica
    concurrence: float = 0.0           # Concurrencia (Acoplamiento)
    kl_divergence: float = 0.0         # Divergencia KL
    inertia: float = 0.0               # Inercia contextual
    n_entangled_gates: int = 0         # Enlaces acoplados activos
    annealing_temp: float = 0.95       # Temperatura de Exploración
    
    continuity: float = 1.0
    state_continuity: float = 1.0
    dominant_dim: str = ""
    dominant_index: int = 0
    coupling_activation: float = 0.0
    break_detected: bool = False

    # Vectores y listas para Dashboard
    psi: List[float] = field(default_factory=list)
    probs: List[float] = field(default_factory=list)
    semantic_weights: dict = field(default_factory=dict)
    active_qubits: List[int] = field(default_factory=list)
    multiverse_states: List[List] = field(default_factory=list) # [[label, prob], ...]
    entangled_names: List[List[str]] = field(default_factory=list) # [[a, b], ...]
    constructive_pairs: List[List[str]] = field(default_factory=list)
    destructive_pairs: List[List[str]] = field(default_factory=list)
    top_dims: List[Tuple[str, float]] = field(default_factory=list)

    # Decisiones generativas
    directive: str = ""
    directive_label: str = "equilibrada"
    collapse_type: str = "colapso_directo"
    collapse_script: str = ""
    n_perspectives: int = 2
    structure_label: str = ""
    context_drift: str = "continuo"

    # ML
    ml_label: str = "general"
    ml_confidence: float = 0.0

    # Green AI & CodeCarbon Hardware Telemetry
    emissions_gco2: float = 0.0
    total_emissions_gco2: float = 0.0
    cpu_power_w: float = 0.0
    ram_power_w: float = 20.0
    total_power_w: float = 0.0
    cpu_energy_kwh: float = 0.0
    ram_energy_kwh: float = 0.0
    total_energy_kwh: float = 0.0
    total_energy_kwh_acc: float = 0.0
    water_consumed_l: float = 0.0
    water_consumed_l_acc: float = 0.0
    ollama_model: str = "qwen2.5:1.5b"

    # Hiperparámetros
    alpha: float = ALPHA_DEFAULT
    coupling: float = COUPLING_DEFAULT
    theta_quiebre: float = THETA_QUIEBRE


# ══════════════════════════════════════════════════════════════════════════════
#  CAPA 1 — CLASIFICADOR DE INTENCIÓN (TF-IDF + LR)
# ══════════════════════════════════════════════════════════════════════════════
class IntentClassifier:
    LABELS = ["tecnologia", "trabajo", "estudio", "bienestar", "conversacional", "general"]
    _MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_v8.pkl")

    _CORPUS = [
        # tecnologia
        ("tengo un bug en python","tecnologia"),("como hago un for loop en python","tecnologia"),
        ("error en mi sql query","tecnologia"),("necesito hacer una api en flask","tecnologia"),
        ("como usar pandas para filtrar datos","tecnologia"),("me da error de importacion","tecnologia"),
        ("como optimizar esta consulta sql","tecnologia"),("la funcion no retorna el valor correcto","tecnologia"),
        ("como conectar python con postgresql","tecnologia"),("tengo un null pointer en java","tecnologia"),
        ("como usar docker para mi proyecto","tecnologia"),("git me dice que hay un conflicto de merge","tecnologia"),
        ("como hacer un join entre dos tablas","tecnologia"),("necesito implementar autenticacion jwt","tecnologia"),
        ("el endpoint de mi api no responde","tecnologia"),("como hacer una clase con herencia en python","tecnologia"),
        ("el programa no compila","tecnologia"),("como usar numpy para multiplicar matrices","tecnologia"),
        ("me tira un keyerror en el diccionario","tecnologia"),("como instalar una libreria con pip","tecnologia"),
        ("tengo un memory leak en mi script python","tecnologia"),("como serializar un objeto en json python","tecnologia"),
        ("el loop infinito no para necesito debugging","tecnologia"),("como usar react hooks en mi componente","tecnologia"),
        ("cual es la diferencia entre inner y left join","tecnologia"),("como hacer un insert en postgresql","tecnologia"),
        ("mi aplicacion nodejs no arranca","tecnologia"),("como usar asyncio en python","tecnologia"),
        ("el despliegue falla con error","tecnologia"),("como debuggear un script de python","tecnologia"),
        ("necesito refactorizar este codigo de sql","tecnologia"),("como crear un indice en mysql","tecnologia"),
        ("la query esta muy lenta necesito optimizar","tecnologia"),("como usar git rebase","tecnologia"),
        ("hay un error de sintaxis en mi codigo","tecnologia"),("como desplegar en docker compose","tecnologia"),
        ("el servidor no encuentra el modulo","tecnologia"),("como usar tensorflow para clasificacion","tecnologia"),
        ("necesito hacer un script de automatizacion","tecnologia"),("como parsear json en javascript","tecnologia"),
        ("el api rest devuelve 500","tecnologia"),("como hacer pruebas unitarias en python","tecnologia"),
        ("cual es la diferencia entre list y tuple","tecnologia"),("como manejar excepciones en python","tecnologia"),
        ("necesito una expresion regular para validar email","tecnologia"),("como usar context manager en python","tecnologia"),
        ("hay una condicion de carrera en mi codigo","tecnologia"),("como implementar un algoritmo de busqueda","tecnologia"),
        ("el codigo de hola mundo en python","tecnologia"),("como leer un archivo csv con pandas","tecnologia"),
        ("dame el codigo de hola mundo en python","tecnologia"),
        # trabajo
        ("tengo demasiado trabajo acumulado","trabajo"),("mi jefe me pide el informe para hoy","trabajo"),
        ("el proyecto esta retrasado","trabajo"),("tengo una reunion importante manana","trabajo"),
        ("no se como organizar mis tareas del dia","trabajo"),("hay un conflicto en el equipo de trabajo","trabajo"),
        ("me asignaron una tarea urgente","trabajo"),("necesito hacer una presentacion para el cliente","trabajo"),
        ("el deadline es esta semana","trabajo"),("mi empresa quiere que cambie de area","trabajo"),
        ("como delegar tareas a mi equipo","trabajo"),("tengo demasiadas prioridades al mismo tiempo","trabajo"),
        ("el sprint termina manana y tenemos pendientes","trabajo"),("me pidieron un plan de accion urgente","trabajo"),
        ("como lidiar con un companero dificil","trabajo"),("necesito preparar un reporte de avance","trabajo"),
        ("el cliente no aprueba el entregable","trabajo"),("tengo que hacer una propuesta de proyecto","trabajo"),
        ("el equipo no cumple los plazos","trabajo"),("me promovieron y no se como manejar el equipo","trabajo"),
        ("necesito un plan para terminar el proyecto","trabajo"),("hay demasiadas reuniones y no avanzo","trabajo"),
        ("el presupuesto del proyecto se agoto","trabajo"),("como hacer una buena presentacion ejecutiva","trabajo"),
        ("mi jefe me critico frente al equipo","trabajo"),("como negociar un aumento de sueldo","trabajo"),
        ("necesito delegar pero no confio en el equipo","trabajo"),("el proyecto tiene requerimientos cambiantes","trabajo"),
        ("como priorizar cuando todo es urgente","trabajo"),("tengo que entregar el proyecto en 3 dias","trabajo"),
        # estudio
        ("necesito aprender python desde cero","estudio"),("como estudiar para un examen de matematicas","estudio"),
        ("no entiendo la teoria de grafos","estudio"),("recomiendame recursos para aprender sql","estudio"),
        ("como practicar programacion todos los dias","estudio"),("tengo un parcial de algoritmos la semana proxima","estudio"),
        ("cual es la mejor forma de aprender machine learning","estudio"),("no entiendo los punteros en c","estudio"),
        ("como memorizar formulas de estadistica","estudio"),("me puedes explicar que es la recursion","estudio"),
        ("quiero entender como funcionan las redes neuronales","estudio"),("hay algun tutorial para principiantes","estudio"),
        ("como empezar a estudiar inteligencia artificial","estudio"),("necesito un resumen de complejidad algoritmica","estudio"),
        ("como aprender calculo diferencial rapido","estudio"),("que libros recomiendas para aprender programacion","estudio"),
        ("explicame que es programacion orientada a objetos","estudio"),("no entiendo la notacion big o","estudio"),
        ("como estudiar sin distraerse","estudio"),("cual es la diferencia entre supervisado y no supervisado","estudio"),
        ("necesito practicar ejercicios de algebra lineal","estudio"),("como entender los arboles de decision","estudio"),
        ("me cuesta entender la recursion en python","estudio"),("explicame las redes convolucionales","estudio"),
        ("que es el gradiente descendente","estudio"),("como prepararse para una entrevista tecnica","estudio"),
        ("necesito aprender react desde cero","estudio"),("donde puedo practicar sql con ejercicios","estudio"),
        ("como estudiar ciberseguridad","estudio"),("que es la entropia en teoria de la informacion","estudio"),
        # bienestar
        ("me siento muy estresado con el trabajo","bienestar"),("no puedo dormir por la ansiedad","bienestar"),
        ("estoy agotado mentalmente","bienestar"),("me siento frustrado y no se que hacer","bienestar"),
        ("tengo mucho miedo de fracasar","bienestar"),("estoy muy cansado ultimamente","bienestar"),
        ("me cuesta mucho concentrarme en el trabajo","bienestar"),("no me siento motivado para nada","bienestar"),
        ("me siento solo y sin energia","bienestar"),("el estres me tiene completamente bloqueado","bienestar"),
        ("como manejar la ansiedad en el trabajo","bienestar"),("me siento abrumado con todo","bienestar"),
        ("necesito un descanso pero no puedo parar","bienestar"),("estoy al limite de mis fuerzas","bienestar"),
        ("me da mucha angustia pensar en el futuro","bienestar"),("llevo semanas sin dormir bien","bienestar"),
        ("siento que no soy suficientemente bueno","bienestar"),("me cuesta levantarme por las mananas","bienestar"),
        ("no encuentro sentido en lo que hago","bienestar"),("me siento incomprendido en el trabajo","bienestar"),
        ("tengo ataques de panico antes de las reuniones","bienestar"),("estoy triste sin saber por que","bienestar"),
        ("me da miedo pedir ayuda","bienestar"),("siento que todo me supera","bienestar"),
        ("como recuperar la motivacion","bienestar"),("me siento paralizado por el perfeccionismo","bienestar"),
        ("lloro sin motivo aparente","bienestar"),("no tengo ganas de hacer nada","bienestar"),
        ("me siento vacio por dentro","bienestar"),("el agotamiento no me deja pensar bien","bienestar"),
        # conversacional
        ("hola","conversacional"),("buenas","conversacional"),("buenos dias","conversacional"),
        ("buenas tardes","conversacional"),("como estas hoy","conversacional"),("que tal estas","conversacional"),
        ("todo bien por ahi","conversacional"),("quien eres tu","conversacional"),("que eres exactamente","conversacional"),
        ("para que sirves","conversacional"),("que puedes hacer por mi","conversacional"),("gracias por tu ayuda","conversacional"),
        ("muchas gracias","conversacional"),("ok entendido muchas gracias","conversacional"),("cuantos anos tienes","conversacional"),
        ("eres un robot o una ia","conversacional"),("tienes sentimientos reales","conversacional"),("eres muy inteligente","conversacional"),
        ("hola que tal todo bien","conversacional"),("nos vemos luego","conversacional"),("chao hasta luego","conversacional"),
        ("hasta pronto","conversacional"),("adios","conversacional"),("genial gracias por todo","conversacional"),
        ("perfecto muchas gracias","conversacional"),("eso estuvo muy bien explicado","conversacional"),
        ("me alegra que me hayas ayudado","conversacional"),("eres util","conversacional"),("que version eres","conversacional"),
        ("tienes nombre","conversacional"),("cuanto sabes","conversacional"),("eres mejor que otros asistentes","conversacional"),
        ("me puedes ayudar con algo","conversacional"),("hola bot","conversacional"),("hey","conversacional"),
        # general
        ("que opinas sobre la vida y la muerte","general"),("cual es el sentido de la existencia humana","general"),
        ("crees en dios","general"),("que es la felicidad para ti","general"),("es posible ser completamente feliz","general"),
        ("que piensas del futuro de la humanidad","general"),("cual es tu opinion sobre la inteligencia artificial","general"),
        ("hay algo que no sepas responder","general"),("cual es la pregunta mas dificil del universo","general"),
        ("que es mas importante el dinero o la felicidad","general"),("como ves el mundo en 10 anos","general"),
        ("crees que los robots van a reemplazar a los humanos","general"),("que es el libre albedrio","general"),
        ("existe la verdad objetiva","general"),("como defines el amor","general"),
        ("que es mas importante la logica o la intuicion","general"),("el universo tiene un proposito","general"),
        ("que piensas de la muerte","general"),("es la consciencia solo quimica cerebral","general"),
        ("como sabes que eres real","general"),("que es el tiempo para ti","general"),
        ("puede una maquina ser creativa de verdad","general"),("cual es la capital de francia","general"),
        ("por que el cielo es azul","general"),("como funciona el sistema solar","general"),
        ("que piensas de la vida y la muerte a nivel cuantico","general")
    ]

    def __init__(self):
        self.pipeline = None
        self.available = _ML_AVAILABLE
        if self.available:
            self._load_or_train()

    @staticmethod
    def normalize(text: str) -> str:
        t = unicodedata.normalize("NFKD", text.strip().lower())
        return "".join(c for c in t if not unicodedata.combining(c))

    def _build(self):
        return Pipeline([
            ("features", FeatureUnion([
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=6000, sublinear_tf=True)),
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=4000, sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=3.0, max_iter=500, class_weight="balanced", random_state=42, solver="lbfgs")),
        ])

    def _load_or_train(self):
        if os.path.exists(self._MODEL_FILE):
            try:
                self.pipeline = joblib.load(self._MODEL_FILE)
                return
            except Exception:
                pass
        self.train()

    def train(self) -> str:
        texts, labels = zip(*self._CORPUS)
        self.pipeline = self._build()
        self.pipeline.fit([self.normalize(t) for t in texts], list(labels))
        try:
            joblib.dump(self.pipeline, self._MODEL_FILE)
        except OSError:
            pass
        return f"Clasificador entrenado con {len(self._CORPUS)} ejemplos."

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        if not self.available or self.pipeline is None:
            return "general", 0.0, {}
        try:
            probs = self.pipeline.predict_proba([self.normalize(text)])[0]
            classes = list(self.pipeline.classes_)
            idx = int(np.argmax(probs))
            return classes[idx], float(probs[idx]), {c: float(p) for c, p in zip(classes, probs)}
        except Exception:
            return "general", 0.0, {}

    @property
    def corpus_size(self) -> int:
        return len(self._CORPUS)


# ══════════════════════════════════════════════════════════════════════════════
#  CAPA 2A — CODIFICADOR SEMÁNTICO
# ══════════════════════════════════════════════════════════════════════════════
class SemanticEncoder:
    def __init__(self):
        self._kw = {i: [IntentClassifier.normalize(k) for k in kws]
                    for i, (_, _, kws) in DIM_SEMANTICS.items()}

    def encode(self, text: str, intent_probs: Dict[str, float]) -> Tuple[np.ndarray, Dict[int, float]]:
        clean = IntentClassifier.normalize(text)
        words = set(re.findall(r"[a-z0-9_+#.-]{2,}", clean))

        raw = np.zeros(N_DIMS)
        coverage: Dict[int, float] = {}
        for i, kws in self._kw.items():
            hits = sum(1 for k in kws if (" " in k and k in clean) or (" " not in k and k in words))
            cov = hits / max(len(kws), 1)
            coverage[i] = cov
            raw[i] = cov

        for label, p in (intent_probs or {}).items():
            for d in INTENT_TO_DIMS.get(label, []):
                raw[d] += p * ML_PROJECTION_WEIGHT

        return raw, coverage


# ══════════════════════════════════════════════════════════════════════════════
#  CAPA 2B — MOTOR DE CONTEXTO CUÁNTICO 12D
# ══════════════════════════════════════════════════════════════════════════════
class ContextEngine:
    def __init__(self, alpha=ALPHA_DEFAULT, coupling=COUPLING_DEFAULT,
                 theta_quiebre=THETA_QUIEBRE, seed: Optional[int] = None):
        self.alpha = alpha
        self.coupling = coupling
        self.theta_quiebre = theta_quiebre
        self.C = COUPLING_MATRIX
        self.psi_prev: Optional[np.ndarray] = None
        self.probs_prev: Optional[np.ndarray] = None
        self.turn = 0
        self.same_dim_turns = 0
        self.last_dominant = -1
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.psi_prev = None
        self.probs_prev = None
        self.turn = 0
        self.same_dim_turns = 0
        self.last_dominant = -1

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else np.ones(N_DIMS) / np.sqrt(N_DIMS)

    def step(self, psi_raw: np.ndarray) -> dict:
        self.turn += 1
        phi = self._norm(psi_raw)

        # 1. Fidelidad (Similitud Coseno)
        if self.psi_prev is None:
            continuity = 1.0
            break_detected = False
        else:
            continuity = float(np.clip(phi @ self.psi_prev, 0.0, 1.0))
            break_detected = continuity < self.theta_quiebre

        # 2. Entropía  Shannon (Input puro)
        p_input = phi ** 2
        p_input = p_input / max(p_input.sum(), 1e-12)
        entropy = float(-(p_input * np.log2(p_input + 1e-12)).sum() / np.log2(N_DIMS))
        entropy = float(np.clip(entropy, 0.0, 1.0))

        # 3. Estabilización EWMA
        if self.psi_prev is None or break_detected:
            psi = phi.copy()
        else:
            psi = self.alpha * phi + (1.0 - self.alpha) * self.psi_prev
        psi = self._norm(psi)

        psi_before_coupling = psi.copy()

        # 4. Acoplamiento Semántico
        psi = (1.0 - self.coupling) * psi + self.coupling * (self.C @ psi)
        psi = self._norm(psi)
        coupling_activation = float(np.clip(1.0 - abs(psi_before_coupling @ psi), 0.0, 1.0))

        # 5. Medición Proyectiva
        p_state = psi ** 2
        p_state = p_state / max(p_state.sum(), 1e-12)
        k = int(np.argmax(p_state))

        if k == self.last_dominant:
            self.same_dim_turns += 1
        else:
            self.same_dim_turns = 1
            self.last_dominant = k

        inertia = min(1.0, self.same_dim_turns / 5.0)

        # Concurrencia (grado de acoplamiento entre dimensiones acopladas)
        conc = 0.0
        n_pairs = 0
        for i, j, w in COUPLING_LINKS:
            prod = np.sqrt(p_state[i] * p_state[j])
            if prod > 0.02:
                conc += w * prod
                n_pairs += 1
        concurrence = float(np.clip(conc * 2.0, 0.0, 1.0))

        # Divergencia KL vs Distribución Uniforme (1/12)
        p_unif = 1.0 / N_DIMS
        kl = float(np.sum(p_state * np.log2((p_state + 1e-12) / p_unif)))

        # Annealing Temp
        annealing_temp = float(0.95 * np.exp(-0.07 * (self.turn - 1)))

        # Interferencia Constructiva / Destructiva
        constructive_pairs = []
        destructive_pairs = []
        entangled_names = []
        for i, j, w in COUPLING_LINKS:
            if p_state[i] > 0.03 and p_state[j] > 0.03:
                name_i, name_j = DIM_SEMANTICS[i][0], DIM_SEMANTICS[j][0]
                entangled_names.append([name_i, name_j])
                if self.probs_prev is not None:
                    prev_prod = np.sqrt(self.probs_prev[i] * self.probs_prev[j])
                    curr_prod = np.sqrt(p_state[i] * p_state[j])
                    if curr_prod > prev_prod + 0.01:
                        constructive_pairs.append([name_i, name_j])
                    elif curr_prod < prev_prod - 0.01:
                        destructive_pairs.append([name_i, name_j])

        state_continuity = (1.0 if self.psi_prev is None
                            else float(np.clip(psi @ self.psi_prev, 0.0, 1.0)))

        self.psi_prev = psi.copy()
        self.probs_prev = p_state.copy()

        return {
            "psi": psi,
            "probs": p_state,
            "entropy": entropy,
            "continuity": continuity,
            "state_continuity": state_continuity,
            "dominant_index": k,
            "break_detected": break_detected,
            "coupling_activation": coupling_activation,
            "concurrence": concurrence,
            "kl_divergence": kl,
            "inertia": inertia,
            "n_entangled_gates": n_pairs,
            "annealing_temp": annealing_temp,
            "entangled_names": entangled_names,
            "constructive_pairs": constructive_pairs,
            "destructive_pairs": destructive_pairs,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  DIRECTOR DE GENERACIÓN Y GUIONES NARRATIVOS
# ══════════════════════════════════════════════════════════════════════════════
class ResponseDirector:
    SCRIPTS = {
        "primer_turno":                 "⟨primer turno⟩ Estado semántico inicializado.",
        "colapso_directo":              "⟨resolución directa⟩ Baja entropía semántica. Respuesta enfocada.",
        "superposicion_alta":           "⟨exploración multidimensional⟩ Alta entropía (múltiples dimensiones activas).",
        "ruptura_semantica":            "⟨ruptura semántica⟩ Transición brusca de contexto detectada.",
        "decoherencia":                 "⟨atenuación⟩ Alta inercia acumulada. Reorientando flujo.",
        "entropy_flush":                "⟨entropy flush⟩ Estímulo corto; manteniendo coherencia prev.",
        "acoplamiento_detectado":       "⟨acoplamiento⟩ Fuertes interacciones entre dimensiones.",
        "entrelazamiento_detectado":    "⟨acoplamiento⟩ Fuertes interacciones entre dimensiones.",
        "interferencia_constructiva":   "⟨reforzamiento asociativo⟩ Amplificación de dimensiones comunes.",
        "interferencia_destructiva":    "⟨atenuación contradictoria⟩ Atenuación de frecuencias contradictorias.",
        "colapso_martillo":             "⟨resolución forzada⟩ Seleccionando dimensión principal ante indecisión.",
        "conversacional":               "⟨conversacional⟩ Flujo de interacción estándar."
    }

    def direct(self, cs: ContextSignal) -> ContextSignal:
        h = cs.entropy_vn
        sc = cs.continuity

        # Determinar tipo de directiva semántica
        if cs.turn_number == 1:
            ct = "primer_turno"
        elif cs.break_detected:
            ct = "ruptura_semantica"
        elif cs.inertia >= 0.8:
            ct = "decoherencia"
        elif cs.concurrence >= 0.45:
            ct = "acoplamiento_detectado"
        elif len(cs.constructive_pairs) > 0:
            ct = "interferencia_constructiva"
        elif h < THETA_BAJO:
            ct = "colapso_directo"
        elif h > THETA_ALTO:
            ct = "superposicion_alta"
        else:
            ct = "colapso_directo"

        cs.collapse_type = ct
        cs.collapse_script = self.SCRIPTS.get(ct, f"⟨{ct}⟩ Resolución semántica ejecutada.")

        if h < THETA_BAJO:
            cs.directive_label = "directa"; cs.n_perspectives = 1
            cs.structure_label = "resolución directa"
        elif h <= THETA_ALTO:
            cs.directive_label = "equilibrada"; cs.n_perspectives = 2
            cs.structure_label = "perspectiva dual"
        else:
            cs.directive_label = "exploratoria"; cs.n_perspectives = 3
            cs.structure_label = "perspectiva múltiple"

        if sc >= 0.68:   cs.context_drift = "continuo"
        elif sc >= 0.38: cs.context_drift = "transicion"
        else:           cs.context_drift = "ruptura"

        instr = {
            "directa":     "Responde de forma técnica, directa y concisa.",
            "equilibrada": "Responde de forma clara y bien fundamentada.",
            "exploratoria":"Responde de forma reflexiva, explorando varios aspectos.",
        }[cs.directive_label]

        foco = f" Foco central: {cs.dominant_dim}." if cs.dominant_dim else ""
        quiebre = " El usuario cambió de tema: concéntrate solo en la nueva pregunta." if cs.break_detected else ""
        cs.directive = instr + foco + quiebre
        return cs


# ══════════════════════════════════════════════════════════════════════════════
#  ORQUESTADOR CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
class HybridChatbot:
    BASELINE_SYSTEM_PROMPT = "Eres un asistente útil y amigable en español."

    def __init__(self, llm_enabled=True, ollama_model="qwen2.5:1.5b",
                 alpha=ALPHA_DEFAULT, coupling=COUPLING_DEFAULT,
                 theta_quiebre=THETA_QUIEBRE, seed: Optional[int] = None):
        self.llm_enabled = llm_enabled
        self.mode = "rapido"
        self.ollama_model = ollama_model
        self.ollama_base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        self.max_llm_latency_sec = 60
        self.last_llm_error = ""
        self.ollama_available = False

        self.classifier = IntentClassifier()
        self.encoder = SemanticEncoder()
        self.engine = ContextEngine(alpha, coupling, theta_quiebre, seed)
        self.director = ResponseDirector()

        self.turn_memory = deque(maxlen=6)
        self.last_signal: Optional[ContextSignal] = None
        self.total_emissions_gco2 = 0.0
        self.total_energy_kwh_acc = 0.0
        self.total_water_l_acc = 0.0
        self._last_tracker_emissions = 0.0

        self.tracker = None
        if _CC_AVAILABLE:
            try:
                self.tracker = EmissionsTracker(
                    project_name="quantum_hybrid",
                    save_to_file=False,
                    log_level="error"
                )
                self.tracker.start()
            except Exception:
                self.tracker = None

        here = os.path.dirname(os.path.abspath(__file__))
        self.trace_file = os.path.join(here, "trace_v8.jsonl")

        self._check_ollama()

    def _check_ollama(self):
        try:
            with httpx.Client(timeout=3) as c:
                c.get(self.ollama_base_url.rstrip("/") + "/api/version").raise_for_status()
            self.ollama_available = True
        except Exception as ex:
            self.ollama_available = False
            self.last_llm_error = str(ex)

    def reset(self):
        self.engine.reset()
        self.turn_memory.clear()
        self.last_signal = None

    def _call_ollama(self, messages: list, options: dict) -> Optional[str]:
        if not (self.llm_enabled and self.ollama_available):
            self.last_llm_error = "Ollama no disponible o deshabilitado."
            return None
        try:
            with httpx.Client(timeout=httpx.Timeout(self.max_llm_latency_sec, connect=4)) as c:
                r = c.post(self.ollama_base_url.rstrip("/") + "/api/chat",
                           json={"model": self.ollama_model, "messages": messages,
                                 "stream": False, "options": options})
                r.raise_for_status()
                content = r.json().get("message", {}).get("content", "").strip()
            if content:
                self.last_llm_error = ""
                return content
            self.last_llm_error = "Respuesta vacía de Ollama."
        except httpx.TimeoutException:
            self.last_llm_error = f"Timeout ({self.max_llm_latency_sec}s)."
        except Exception as ex:
            self.last_llm_error = str(ex)
        return None

    def _generate_classical_fallback(self, user_input: str, cs: ContextSignal) -> str:
        clean = IntentClassifier.normalize(user_input)
        
        # Comprobaciones rápidas directas
        if "hola mundo" in clean and "python" in clean:
            return "¡Claro! Aquí tienes el código de 'Hola Mundo' en Python:\n\n```python\nprint(\"Hola Mundo\")\n```\n\nEste es el punto de entrada clásico para comprobar la ejecución de scripts en Python."
        
        if "vida y la muerte" in clean or "semantico" in clean or "cuantico" in clean:
            return "En términos semánticos y de teoría de la información, el concepto de 'vida' y 'muerte' puede ser interpretado desde la distribución multidimensional de probabilidades y la teoría de sistemas complejos. Un sistema de información permanece en una configuración de probabilidades contextuales hasta que interactúa con un entorno o consulta, convergiendo hacia una resolución determinada. Desde este enfoque, la información latente se acopla con el espacio temático circundante."

        if cs.ml_label == "conversacional" or clean in ["hola", "buenas", "buenos dias"]:
            return "¡Hola! ¿Cómo puedo ayudarte hoy? Estoy listo para responder tus preguntas técnicas, filosóficas o de desarrollo."

        if cs.ml_label == "tecnologia":
            return f"He procesado tu consulta técnica focalizada en **{cs.dominant_dim}**. El sistema detecta un nivel de entropía H={cs.entropy_vn:.2f}. ¿Deseas un ejemplo de código específico o una explicación conceptual de la arquitectura?"

        return f"Procesando estímulo en dimensión semántica **{cs.dominant_dim}** (Entropía H={cs.entropy_vn:.2f}, Continuidad={cs.continuity:.2f}). ¿Podrías detallar más tu inquietud para profundizar el análisis?"

    def respond(self, user_input: str) -> dict:
        t0 = time.perf_counter()

        # Capa 1
        t_cls = time.perf_counter()
        label, conf, probs = self.classifier.predict(user_input)
        cls_ms = (time.perf_counter() - t_cls) * 1000.0

        # Capa 2
        t_eng = time.perf_counter()
        psi_raw, coverage = self.encoder.encode(user_input, probs)
        res = self.engine.step(psi_raw)
        eng_ms = (time.perf_counter() - t_eng) * 1000.0

        # Estructurar ContextSignal
        cs = ContextSignal(
            turn_number=self.engine.turn,
            mode=self.mode,
            entropy_vn=round(res["entropy"], 4),
            fidelity=round(res["continuity"], 4),
            concurrence=round(res["concurrence"], 4),
            kl_divergence=round(res["kl_divergence"], 4),
            inertia=round(res["inertia"], 4),
            n_entangled_gates=res["n_entangled_gates"],
            annealing_temp=round(res["annealing_temp"], 4),
            continuity=round(res["continuity"], 4),
            state_continuity=round(res["state_continuity"], 4),
            dominant_index=res["dominant_index"],
            dominant_dim=DIM_SEMANTICS[res["dominant_index"]][0],
            coupling_activation=round(res["coupling_activation"], 4),
            break_detected=res["break_detected"],
            psi=[round(float(x), 4) for x in res["psi"]],
            probs=[round(float(x), 4) for x in res["probs"]],
            semantic_weights={DIM_SEMANTICS[i][0]: round(float(coverage.get(i, 0.0)), 3)
                              for i in range(N_DIMS)},
            active_qubits=[i for i in range(N_DIMS) if coverage.get(i, 0.0) > 0.0 or res["probs"][i] > 0.05],
            entangled_names=res["entangled_names"],
            constructive_pairs=res["constructive_pairs"],
            destructive_pairs=res["destructive_pairs"],
            ml_label=label,
            ml_confidence=round(conf, 4),
            alpha=self.engine.alpha,
            coupling=self.engine.coupling,
            theta_quiebre=self.engine.theta_quiebre,
        )

        # Multiverso
        order = np.argsort(-res["probs"])
        cs.multiverse_states = [[DIM_SEMANTICS[int(i)][0], round(float(res["probs"][int(i)]), 4)] for i in order[:3]]
        cs.top_dims = [(DIM_SEMANTICS[int(i)][0], round(float(res["probs"][int(i)]), 4)) for i in order[:3]]

        # Director
        cs = self.director.direct(cs)

        # Capa 3 Generación
        history = []
        for u, b in list(self.turn_memory)[-2:]:
            history.append({"role": "user", "content": u[:200]})
            history.append({"role": "assistant", "content": b[:200]})

        system = (f"Eres un asistente conversacional en español. {cs.directive} "
                  f"No inventes librerías ni datos ficticios.")
        messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_input}]

        temp = max(0.1, min(0.9, cs.annealing_temp))
        options = {"temperature": temp, "top_p": 0.9, "num_predict": 512, "num_ctx": 2048}

        t_llm = time.perf_counter()
        text = self._call_ollama(messages, options)
        llm_ms = (time.perf_counter() - t_llm) * 1000.0

        if text is None:
            text = self._generate_classical_fallback(user_input, cs)
            source = "classical"
        else:
            source = "ollama"

        total_ms = (time.perf_counter() - t0) * 1000.0
        exec_sec = total_ms / 1000.0

        # Base hardware power (Watts)
        ram_w = 20.0  # Standard CodeCarbon RAM Power
        cpu_w = round(min(40.0, max(4.0, 9.5 + min(25.0, exec_sec * 1.5))), 2)

        if self.tracker:
            try:
                if hasattr(self.tracker, "_last_measured_power"):
                    pow_obj = self.tracker._last_measured_power
                    if hasattr(pow_obj, "cpu_power") and pow_obj.cpu_power:
                        cpu_w = round(float(pow_obj.cpu_power), 2)
                    if hasattr(pow_obj, "ram_power") and pow_obj.ram_power:
                        ram_w = round(float(pow_obj.ram_power), 2)
            except Exception:
                pass

        total_w = round(ram_w + cpu_w, 2)

        # Energy calculations in kWh
        ram_kwh = round((ram_w * exec_sec) / (3600.0 * 1000.0), 8)
        cpu_kwh = round((cpu_w * exec_sec) / (3600.0 * 1000.0), 8)
        total_kwh = round(ram_kwh + cpu_kwh, 8)
        water_l = round(total_kwh * 0.25, 8)

        # Measurement via CodeCarbon or CPU power estimation fallback
        emissions_gco2 = 0.0
        if self.tracker:
            try:
                curr_emissions_kg = self.tracker.flush() or 0.0
                if curr_emissions_kg > 0:
                    delta_kg = curr_emissions_kg - self._last_tracker_emissions
                    self._last_tracker_emissions = curr_emissions_kg
                    emissions_gco2 = round(max(0.0, delta_kg) * 1000.0, 5)
            except Exception:
                emissions_gco2 = 0.0

        if emissions_gco2 <= 0:
            emissions_gco2 = round(total_kwh * 161.6, 5)

        self.total_emissions_gco2 = round(self.total_emissions_gco2 + emissions_gco2, 5)
        self.total_energy_kwh_acc = round(self.total_energy_kwh_acc + total_kwh, 8)
        self.total_water_l_acc = round(self.total_water_l_acc + water_l, 8)

        cs.emissions_gco2 = emissions_gco2
        cs.total_emissions_gco2 = self.total_emissions_gco2
        cs.cpu_power_w = cpu_w
        cs.ram_power_w = ram_w
        cs.total_power_w = total_w
        cs.cpu_energy_kwh = cpu_kwh
        cs.ram_energy_kwh = ram_kwh
        cs.total_energy_kwh = total_kwh
        cs.total_energy_kwh_acc = self.total_energy_kwh_acc
        cs.water_consumed_l = water_l
        cs.water_consumed_l_acc = self.total_water_l_acc
        cs.ollama_model = self.ollama_model
        cs.exec_time_ms = round(total_ms, 1)

        self.last_signal = cs
        self.turn_memory.append((user_input, text))

        return {
            "response": text,
            "mode": self.mode,
            "source": source,
            "total_ms": round(total_ms, 1),
            "emissions_gco2": emissions_gco2,
            "total_emissions_gco2": self.total_emissions_gco2,
            "cpu_power_w": cpu_w,
            "ram_power_w": ram_w,
            "total_power_w": total_w,
            "total_energy_kwh": total_kwh,
            "total_energy_kwh_acc": self.total_energy_kwh_acc,
            "water_consumed_l": water_l,
            "water_consumed_l_acc": self.total_water_l_acc,
            "quantum_signal": asdict(cs)
        }




# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML INLINE
# ══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chatbot Híbrido Neuro-Simbólico — Lab Dashboard v8.1</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet">
<style>
/* ── Reset & Variables ─────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg0:       #020409;
  --bg1:       #070d1a;
  --bg2:       #0b1528;
  --bg3:       #0f1e38;
  --panel:     #0d1a2e;
  --panel2:    #111f35;
  --border:    #1a2e50;
  --border2:   #243d66;
  --cyan:      #00f5e4;
  --cyan2:     #00c9d4;
  --cyan-dim:  #006e7a;
  --purple:    #a259ff;
  --purple2:   #7c3aed;
  --purple-dim:#3d1f7a;
  --green:     #39ff7a;
  --green-dim: #0f3d1f;
  --amber:     #ffb830;
  --amber-dim: #4a3400;
  --red:       #ff4566;
  --red-dim:   #3d0015;
  --text:      #c8ddf8;
  --text2:     #7ea8d8;
  --text3:     #4a6a9a;
  --text4:     #2a3e5a;
  --mono:      'Share Tech Mono', monospace;
  --display:   'Orbitron', sans-serif;
  --body:      'Exo 2', sans-serif;
  --glow-c:    0 0 12px rgba(0,245,228,0.4), 0 0 30px rgba(0,245,228,0.15);
  --glow-p:    0 0 12px rgba(162,89,255,0.4), 0 0 30px rgba(162,89,255,0.15);
  --glow-g:    0 0 12px rgba(57,255,122,0.4), 0 0 30px rgba(57,255,122,0.15);
}

html { font-size: 14px; }
body {
  background: var(--bg0);
  color: var(--text);
  font-family: var(--body);
  height: 100vh;
  overflow: hidden;
  display: grid;
  grid-template-rows: 52px 1fr;
  grid-template-columns: 1fr;
}

body::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.07) 2px,
    rgba(0,0,0,0.07) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

header {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 20px;
  gap: 20px;
  position: relative;
  overflow: hidden;
}
header::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: scan-line 3s linear infinite;
}
@keyframes scan-line {
  0%   { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.logo {
  font-family: var(--display);
  font-weight: 900;
  font-size: 0.95rem;
  color: var(--cyan);
  text-shadow: var(--glow-c);
  letter-spacing: 0.15em;
  white-space: nowrap;
}
.logo span { color: var(--purple); }
.header-tags { display: flex; gap: 8px; flex: 1; }
.tag {
  font-family: var(--mono);
  font-size: 0.62rem;
  padding: 2px 8px;
  border-radius: 2px;
  letter-spacing: 0.08em;
}
.tag-c { background: rgba(0,245,228,0.08); border: 1px solid var(--cyan-dim); color: var(--cyan2); }
.tag-p { background: rgba(162,89,255,0.08); border: 1px solid var(--purple-dim); color: var(--purple); }
.tag-g { background: rgba(57,255,122,0.08); border: 1px solid var(--green-dim); color: var(--green); }
.tag-a { background: rgba(255,184,48,0.08);  border: 1px solid var(--amber-dim); color: var(--amber); }

.header-status {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text3);
  display: flex; align-items: center; gap: 6px;
}
.dot { width: 6px; height: 6px; border-radius: 50%; }
.dot-on  { background: var(--green); box-shadow: var(--glow-g); animation: pulse 2s ease-in-out infinite; }
.dot-off { background: var(--text4); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

.api-url-input {
  font-family: var(--mono);
  font-size: 0.68rem;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border2);
  border-radius: 3px;
  color: var(--cyan2);
  padding: 4px 10px;
  width: 240px;
  outline: none;
}
.api-url-input:focus { border-color: var(--cyan); box-shadow: 0 0 8px rgba(0,245,228,0.2); }

.main {
  display: grid;
  grid-template-columns: 340px 1fr 300px;
  grid-template-rows: 1fr;
  height: 100%;
  overflow: hidden;
  gap: 0;
}

.panel {
  background: var(--panel);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
  overflow: hidden;
}
.panel:last-child { border-right: none; border-left: 1px solid var(--border); }
.panel-header {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  font-family: var(--display);
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  color: var(--text3);
  text-transform: uppercase;
  display: flex; align-items: center; gap: 8px;
  flex-shrink: 0;
}
.panel-header .ph-icon { color: var(--cyan); font-size: 0.8rem; }
.panel-content { flex: 1; overflow-y: auto; padding: 12px; display: flex; flex-direction: column; gap: 10px; }
.panel-content::-webkit-scrollbar { width: 4px; }
.panel-content::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.chat-panel {
  display: flex; flex-direction: column;
  background: var(--bg1);
  overflow: hidden;
}
.chat-messages {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 10px;
  scroll-behavior: smooth;
}
.chat-messages::-webkit-scrollbar { width: 4px; }
.chat-messages::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.msg {
  display: flex; gap: 10px; align-items: flex-start;
  animation: msg-in 0.3s ease both;
}
@keyframes msg-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }

.msg-user { flex-direction: row-reverse; }
.msg-avatar {
  width: 28px; height: 28px; border-radius: 4px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--display); font-size: 0.55rem; font-weight: 700;
}
.avatar-user { background: var(--purple-dim); border: 1px solid var(--purple); color: var(--purple); }
.avatar-bot  { background: rgba(0,245,228,0.08); border: 1px solid var(--cyan-dim); color: var(--cyan); }

.msg-bubble {
  max-width: 80%;
  padding: 10px 13px;
  border-radius: 6px;
  font-size: 0.82rem;
  line-height: 1.55;
}
.bubble-user {
  background: rgba(162,89,255,0.1);
  border: 1px solid rgba(162,89,255,0.3);
  color: var(--text);
}
.bubble-bot {
  background: rgba(0,245,228,0.05);
  border: 1px solid rgba(0,245,228,0.15);
  color: var(--text);
}
.bubble-bot code {
  font-family: var(--mono);
  font-size: 0.75rem;
  background: rgba(0,0,0,0.4);
  padding: 0 4px;
  border-radius: 2px;
  color: var(--cyan2);
}
.bubble-bot pre {
  font-family: var(--mono);
  font-size: 0.72rem;
  background: rgba(0,0,0,0.5);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
  margin: 6px 0;
  overflow-x: auto;
  color: var(--cyan2);
  line-height: 1.5;
}
.msg-meta {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--text4);
  margin-top: 3px;
  display: flex; gap: 8px;
}
.meta-label { color: var(--cyan-dim); }

.collapse-badge {
  font-family: var(--mono);
  font-size: 0.62rem;
  padding: 3px 8px;
  border-radius: 2px;
  margin-bottom: 5px;
  display: inline-block;
  animation: badge-in 0.5s ease both;
}
@keyframes badge-in { from { opacity: 0; } to { opacity: 1; } }
.cb-colapso_directo        { background: rgba(0,245,228,0.08); color: var(--cyan2); border-left: 2px solid var(--cyan); }
.cb-superposicion_alta     { background: rgba(162,89,255,0.1); color: var(--purple); border-left: 2px solid var(--purple); }
.cb-ruptura_semantica      { background: rgba(255,69,102,0.1); color: var(--red); border-left: 2px solid var(--red); }
.cb-decoherencia           { background: rgba(255,184,48,0.1); color: var(--amber); border-left: 2px solid var(--amber); }
.cb-entropy_flush          { background: rgba(0,201,212,0.08); color: var(--cyan2); border-left: 2px solid var(--cyan2); }
.cb-entrelazamiento_detectado, .cb-acoplamiento_detectado { background: rgba(162,89,255,0.12); color: var(--purple); border-left: 2px solid var(--purple); }
.cb-interferencia_constructiva { background: rgba(57,255,122,0.1); color: var(--green); border-left: 2px solid var(--green); }
.cb-interferencia_destructiva  { background: rgba(255,69,102,0.1); color: var(--red); border-left: 2px solid var(--red); }
.cb-colapso_martillo       { background: rgba(255,69,102,0.12); color: var(--red); border-left: 2px solid var(--red); }
.cb-primer_turno           { background: rgba(57,255,122,0.08); color: var(--green); border-left: 2px solid var(--green); }
.cb-convergencia_annealing { background: rgba(255,184,48,0.08); color: var(--amber); border-left: 2px solid var(--amber); }
.cb-conversacional         { background: rgba(7,13,26,0.5); color: var(--text3); border-left: 2px solid var(--border2); }

.chat-input-area {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg2);
  display: flex; gap: 8px; align-items: flex-end;
}
.chat-input {
  flex: 1;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border2);
  border-radius: 4px;
  color: var(--text);
  font-family: var(--body);
  font-size: 0.83rem;
  padding: 9px 13px;
  resize: none;
  outline: none;
  min-height: 40px; max-height: 100px;
  line-height: 1.4;
  transition: border-color 0.2s;
}
.chat-input:focus { border-color: var(--cyan); box-shadow: 0 0 12px rgba(0,245,228,0.15); }
.send-btn {
  background: var(--cyan);
  color: var(--bg0);
  border: none; border-radius: 4px;
  padding: 9px 16px;
  font-family: var(--display);
  font-size: 0.62rem; font-weight: 700;
  letter-spacing: 0.1em;
  cursor: pointer;
  text-transform: uppercase;
  transition: all 0.2s;
  white-space: nowrap;
}
.send-btn:hover { background: var(--cyan2); box-shadow: var(--glow-c); }
.send-btn:disabled { background: var(--border2); color: var(--text4); cursor: not-allowed; box-shadow: none; }

.metric-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
}
.mc-title {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text3);
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 6px;
  display: flex; align-items: center; justify-content: space-between;
}
.mc-title .mc-edu {
  font-size: 0.55rem;
  color: var(--text4);
  cursor: pointer;
  border: 1px solid var(--border);
  border-radius: 2px;
  padding: 1px 5px;
  transition: all 0.15s;
}
.mc-title .mc-edu:hover { color: var(--cyan); border-color: var(--cyan-dim); }

.gauge-wrap { margin-top: 4px; }
.gauge-bar {
  height: 5px;
  border-radius: 3px;
  background: rgba(255,255,255,0.04);
  overflow: hidden;
  position: relative;
}
.gauge-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
  position: relative;
}
.gauge-fill::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 4px; height: 100%;
  background: rgba(255,255,255,0.6);
  border-radius: 2px;
}
.gf-cyan   { background: linear-gradient(90deg, var(--cyan-dim), var(--cyan)); }
.gf-purple { background: linear-gradient(90deg, var(--purple-dim), var(--purple)); }
.gf-green  { background: linear-gradient(90deg, var(--green-dim), var(--green)); }
.gf-amber  { background: linear-gradient(90deg, var(--amber-dim), var(--amber)); }
.gf-red    { background: linear-gradient(90deg, var(--red-dim), var(--red)); }

.gauge-value {
  font-family: var(--mono);
  font-size: 1.0rem;
  font-weight: bold;
  line-height: 1;
  margin-bottom: 3px;
}
.gv-cyan   { color: var(--cyan); text-shadow: var(--glow-c); }
.gv-purple { color: var(--purple); text-shadow: var(--glow-p); }
.gv-green  { color: var(--green); text-shadow: var(--glow-g); }
.gv-amber  { color: var(--amber); }

.gauge-sub {
  font-family: var(--mono);
  font-size: 0.58rem;
  color: var(--text4);
  margin-top: 2px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.mini-metric {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 10px;
}
.mm-label {
  font-family: var(--mono);
  font-size: 0.55rem;
  color: var(--text4);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.mm-val {
  font-family: var(--mono);
  font-size: 0.85rem;
  font-weight: bold;
}

.collapse-display {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
}
.cd-type {
  font-family: var(--mono);
  font-size: 0.62rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 5px;
}
.cd-script {
  font-family: var(--mono);
  font-size: 0.68rem;
  color: var(--text2);
  line-height: 1.5;
  border-left: 2px solid var(--cyan-dim);
  padding-left: 8px;
}

.qubit-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 5px;
}
.qubit-cell {
  aspect-ratio: 1;
  border-radius: 4px;
  border: 1px solid var(--border);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  font-family: var(--mono);
  font-size: 0.5rem;
  cursor: pointer;
  transition: all 0.3s;
  position: relative;
  overflow: hidden;
  padding: 3px;
}
.qubit-cell:hover { border-color: var(--cyan); }
.qubit-cell.active {
  border-color: var(--cyan);
  background: rgba(0,245,228,0.1);
  box-shadow: 0 0 8px rgba(0,245,228,0.2);
}
.qubit-cell.active::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at center, rgba(0,245,228,0.15), transparent 70%);
  animation: qubit-glow 1.5s ease-in-out infinite alternate;
}
@keyframes qubit-glow { from { opacity: 0.5; } to { opacity: 1; } }

.qc-index { color: var(--text4); font-size: 0.45rem; }
.qc-name  { color: var(--text2); font-size: 0.48rem; text-align: center; }
.qubit-cell.active .qc-name { color: var(--cyan); }

.qubit-weight-bar {
  position: absolute; bottom: 0; left: 0;
  height: 2px;
  background: var(--cyan);
  transition: width 0.5s ease;
}

.mv-state {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 8px;
  border-radius: 3px;
  background: rgba(0,0,0,0.2);
  margin-bottom: 4px;
}
.mv-label {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text);
  flex: 1;
}
.mv-bar-wrap { width: 80px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; }
.mv-bar { height: 100%; border-radius: 2px; transition: width 0.5s ease; }
.mv-prob {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text3);
  width: 36px; text-align: right;
}
.mv-rank-0 .mv-bar { background: var(--cyan); }
.mv-rank-1 .mv-bar { background: var(--purple); }
.mv-rank-2 .mv-bar { background: var(--text3); }

.ent-pair {
  display: flex; align-items: center; gap: 6px;
  font-family: var(--mono);
  font-size: 0.65rem;
  padding: 4px 8px;
  border-radius: 3px;
  background: rgba(162,89,255,0.08);
  border: 1px solid rgba(162,89,255,0.2);
  margin-bottom: 4px;
}
.ent-a { color: var(--cyan); }
.ent-tensor { color: var(--text4); font-size: 0.55rem; }
.ent-b { color: var(--purple); }
.ent-type {
  margin-left: auto;
  font-size: 0.55rem;
  padding: 1px 5px;
  border-radius: 2px;
}
.et-constructiva { background: rgba(57,255,122,0.1); color: var(--green); }
.et-destructiva  { background: rgba(255,69,102,0.1);  color: var(--red); }
.et-active       { background: rgba(162,89,255,0.15); color: var(--purple); }

.edu-section {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 10px 12px;
}
.edu-title {
  font-family: var(--display);
  font-size: 0.6rem;
  font-weight: 700;
  letter-spacing: 0.15em;
  color: var(--cyan);
  text-transform: uppercase;
  margin-bottom: 6px;
  display: flex; align-items: center; gap: 6px;
}
.edu-body {
  font-family: var(--body);
  font-size: 0.72rem;
  color: var(--text2);
  line-height: 1.6;
}
.edu-formula {
  font-family: var(--mono);
  font-size: 0.65rem;
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border);
  border-left: 2px solid var(--purple);
  padding: 6px 10px;
  border-radius: 2px;
  margin: 6px 0;
  color: var(--purple);
}
.edu-highlight { color: var(--cyan); font-weight: 600; }
.edu-tag-grid { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.edu-tag {
  font-family: var(--mono);
  font-size: 0.58rem;
  padding: 2px 6px;
  border-radius: 2px;
  background: rgba(0,245,228,0.06);
  border: 1px solid var(--border);
  color: var(--text3);
}

.ml-bar-wrap { margin-top: 6px; }
.ml-row {
  display: flex; align-items: center; gap: 6px;
  margin-bottom: 5px;
}
.ml-name {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text3);
  width: 80px; flex-shrink: 0;
  text-transform: uppercase;
}
.ml-bar-bg {
  flex: 1;
  height: 5px;
  background: rgba(255,255,255,0.04);
  border-radius: 3px;
  overflow: hidden;
}
.ml-bar-fill {
  height: 100%;
  border-radius: 3px;
  transition: width 0.5s cubic-bezier(0.4,0,0.2,1);
  background: linear-gradient(90deg, var(--purple-dim), var(--purple));
}
.ml-bar-fill.ml-active {
  background: linear-gradient(90deg, var(--cyan-dim), var(--cyan));
  box-shadow: 0 0 6px rgba(0,245,228,0.4);
}
.ml-conf {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text4);
  width: 28px; text-align: right;
}

.turn-history { display: flex; flex-direction: column; gap: 4px; }
.turn-item {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 8px;
  border-radius: 3px;
  background: rgba(0,0,0,0.2);
  border: 1px solid transparent;
  font-family: var(--mono);
  font-size: 0.6rem;
  cursor: pointer;
  transition: all 0.15s;
}
.turn-item:hover { border-color: var(--border2); background: var(--bg3); }
.ti-num { color: var(--text4); width: 16px; }
.ti-ent { color: var(--cyan); width: 36px; }
.ti-fid { color: var(--purple); width: 36px; }
.ti-col { color: var(--text3); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ti-source { 
  font-size: 0.52rem;
  padding: 1px 5px;
  border-radius: 2px;
}
.ts-ollama   { background: rgba(57,255,122,0.1);  color: var(--green); }
.ts-classical{ background: rgba(255,184,48,0.1);  color: var(--amber); }
.ts-fallback { background: rgba(255,69,102,0.1);  color: var(--red); }

.ctrl-row {
  display: flex; align-items: center; gap: 8px;
  padding: 5px 0;
  border-bottom: 1px solid var(--border);
}
.ctrl-row:last-child { border-bottom: none; }
.ctrl-label {
  font-family: var(--mono);
  font-size: 0.6rem;
  color: var(--text3);
  width: 90px;
  letter-spacing: 0.08em;
}
.ctrl-val {
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--text);
  flex: 1;
}
.ctrl-btn {
  background: rgba(0,0,0,0.4);
  border: 1px solid var(--border2);
  color: var(--text2);
  border-radius: 2px;
  padding: 3px 8px;
  font-family: var(--mono);
  font-size: 0.6rem;
  cursor: pointer;
  transition: all 0.15s;
}
.ctrl-btn:hover { border-color: var(--cyan); color: var(--cyan); }
.ctrl-btn-on { border-color: var(--green); color: var(--green); }
.ctrl-btn-off { border-color: var(--text4); color: var(--text4); }

.tooltip-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(2,4,9,0.85);
  backdrop-filter: blur(4px);
  z-index: 1000;
  align-items: center;
  justify-content: center;
}
.tooltip-overlay.show { display: flex; }
.tooltip-box {
  background: var(--panel2);
  border: 1px solid var(--cyan);
  border-radius: 6px;
  padding: 24px 28px;
  max-width: 480px;
  width: 90%;
  box-shadow: var(--glow-c);
  animation: tooltip-in 0.25s ease;
}
@keyframes tooltip-in { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: none; } }
.tt-title {
  font-family: var(--display);
  font-size: 0.8rem;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.1em;
  margin-bottom: 12px;
}
.tt-body {
  font-family: var(--body);
  font-size: 0.8rem;
  color: var(--text2);
  line-height: 1.65;
}
.tt-formula {
  font-family: var(--mono);
  font-size: 0.72rem;
  background: rgba(0,0,0,0.4);
  border-left: 2px solid var(--purple);
  padding: 8px 12px;
  margin: 10px 0;
  border-radius: 2px;
  color: var(--purple);
}
.tt-close { margin-top: 16px; text-align: right; }

.thinking-indicator {
  display: flex; align-items: center; gap: 8px;
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--cyan);
  padding: 8px 0;
  animation: msg-in 0.3s ease both;
}
.thinking-dots span {
  display: inline-block;
  width: 5px; height: 5px;
  background: var(--cyan);
  border-radius: 50%;
  animation: dot-bounce 1.2s ease-in-out infinite;
  margin-right: 3px;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dot-bounce { 0%,80%,100% { transform: scale(0.6); opacity: 0.4; } 40% { transform: scale(1); opacity: 1; } }

.sem-chart { display: flex; flex-direction: column; gap: 4px; }
.sem-row { display: flex; align-items: center; gap: 6px; }
.sem-name {
  font-family: var(--mono);
  font-size: 0.55rem;
  color: var(--text3);
  width: 72px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.sem-bar-bg {
  flex: 1; height: 4px;
  background: rgba(255,255,255,0.04);
  border-radius: 2px; overflow: hidden;
}
.sem-bar-fill {
  height: 100%; border-radius: 2px;
  transition: width 0.6s cubic-bezier(0.4,0,0.2,1);
}
.sem-val {
  font-family: var(--mono);
  font-size: 0.55rem;
  color: var(--text4);
  width: 30px; text-align: right;
}

.anneal-gauge { display: flex; align-items: center; gap: 14px; padding: 4px 0; }
.anneal-ring { position: relative; width: 52px; height: 52px; flex-shrink: 0; }
.anneal-ring svg { transform: rotate(-90deg); }
.anneal-ring-bg { fill: none; stroke: rgba(255,255,255,0.05); stroke-width: 4; }
.anneal-ring-fill { fill: none; stroke-width: 4; stroke-linecap: round; transition: stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1); }
.anneal-ring-val {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--mono);
  font-size: 0.65rem;
  color: var(--amber);
}
.anneal-info { font-family: var(--mono); font-size: 0.65rem; color: var(--text3); }
.anneal-info b { color: var(--amber); display: block; font-size: 0.72rem; }
</style>
</head>
<body>

<!-- ── Header ───────────────────────────────────────────────────────── -->
<!-- ── Header ───────────────────────────────────────────────────────── -->
<header>
  <div class="logo">HYBRID<span>BOT</span> LAB <span style="font-size:0.6rem; color:var(--text3)">v8.1 · Neuro-Simbólico 12D</span></div>
  <div class="header-tags">
    <span class="tag tag-c">v8.1-12D</span>
    <span class="tag tag-c" id="tag-params">α=0.3 c=0.5 θ=0.5</span>
    <span class="tag tag-p" id="tag-mode">RAPIDO</span>
    <span class="tag tag-g" id="tag-qubits">12D</span>
    <span class="tag tag-a" id="tag-llm">LLM OFF</span>
    <span class="tag tag-g" id="tag-green">🍃 0.000 gCO₂</span>
  </div>
  <input class="api-url-input" id="apiUrl" value="" placeholder="http://localhost:8080">
  <div class="header-status">
    <div class="dot" id="conn-dot" title="Estado conexión API"></div>
    <span id="conn-label">DESCONECTADO</span>
  </div>
</header>

<!-- ── Main Grid ─────────────────────────────────────────────────────── -->
<div class="main">

  <!-- ═══ LEFT: Semantic State Metrics ═══ -->
  <div class="panel">
    <div class="panel-header">
      <span class="ph-icon">◈</span> MÉTRICAS DEL ESTADO SEMÁNTICO (12D) & GREEN AI
    </div>
    <div class="panel-content" id="left-content">

      <!-- Green AI / CodeCarbon Hardware Tracker -->
      <div class="metric-card" style="border: 1px solid rgba(57,255,122,0.35); background: rgba(15,61,31,0.25)">
        <div class="mc-title" style="color:var(--green)">
          🍃 GREEN AI — CODECARBON TELEMETRY
          <span class="mc-edu" onclick="showEdu('green_ai')">?</span>
        </div>
        <div class="metrics-grid" style="grid-template-columns: 1fr 1fr; gap: 8px;">
          <div>
            <div class="gv-green gauge-value" id="val-co2-turn">0.0000</div>
            <div class="gauge-sub">gCO₂eq / turno</div>
          </div>
          <div>
            <div class="gauge-value" style="color:var(--cyan)" id="val-co2-total">0.0000</div>
            <div class="gauge-sub">gCO₂eq acumulado</div>
          </div>
        </div>
        <div style="margin-top:8px; padding-top:6px; border-top: 1px dashed rgba(57,255,122,0.2); display:grid; grid-template-columns: 1fr 1fr; gap:4px; font-family:var(--mono); font-size:0.58rem; color:var(--text2)">
          <div>⚡ Potencia: <b style="color:var(--amber)" id="val-watts-total">32.5 W</b></div>
          <div>CPU: <span style="color:var(--cyan)" id="val-watts-cpu">12.5 W</span></div>
          <div>RAM: <span style="color:var(--purple)" id="val-watts-ram">20.0 W</span></div>
          <div>💧 Agua: <span style="color:var(--cyan)" id="val-water-l">0.000 L</span></div>
          <div style="grid-column: span 2">🔋 Energía: <span style="color:var(--green)" id="val-kwh-total">0.00000 kWh</span></div>
        </div>
        <div class="gauge-sub" style="margin-top:6px; color:var(--text3); font-size:0.52rem">
          🍃 Factor Ecuador: 0.1616 kgCO₂eq/kWh · Tracking CPU/RAM
        </div>
      </div>


      <!-- Entropy -->
      <div class="metric-card">
        <div class="mc-title">
          ENTROPÍA SEMÁNTICA (Dispersión 12D)
          <span class="mc-edu" onclick="showEdu('entropy')">?</span>
        </div>
        <div class="gv-cyan gauge-value" id="val-entropy">—</div>
        <div class="gauge-wrap">
          <div class="gauge-bar"><div class="gauge-fill gf-cyan" id="bar-entropy" style="width:0%"></div></div>
        </div>
        <div class="gauge-sub" id="sub-entropy">→ perspectivas: —</div>
      </div>

      <!-- Fidelity -->
      <div class="metric-card">
        <div class="mc-title">
          FIDELIDAD SEMÁNTICA (Continuidad Coseno)
          <span class="mc-edu" onclick="showEdu('fidelity')">?</span>
        </div>
        <div class="gv-purple gauge-value" id="val-fidelity">—</div>
        <div class="gauge-wrap">
          <div class="gauge-bar"><div class="gauge-fill gf-purple" id="bar-fidelity" style="width:0%"></div></div>
        </div>
        <div class="gauge-sub" id="sub-fidelity">→ drift: —</div>
      </div>

      <!-- Concurrence + Annealing row -->
      <div class="metrics-grid">
        <div class="mini-metric">
          <div class="mm-label">CONCURRENCIA SEMÁNTICA</div>
          <div class="mm-val gv-cyan" id="val-conc">—</div>
          <div class="gauge-wrap" style="margin-top:5px">
            <div class="gauge-bar"><div class="gauge-fill gf-cyan" id="bar-conc" style="width:0%"></div></div>
          </div>
        </div>
        <div class="mini-metric">
          <div class="mm-label">KL DIVERGENCIA</div>
          <div class="mm-val gv-purple" id="val-kl">—</div>
          <div class="gauge-wrap" style="margin-top:5px">
            <div class="gauge-bar"><div class="gauge-fill gf-purple" id="bar-kl" style="width:0%"></div></div>
          </div>
        </div>
        <div class="mini-metric">
          <div class="mm-label">INERCIA</div>
          <div class="mm-val gv-amber" id="val-inertia">—</div>
          <div class="gauge-wrap" style="margin-top:5px">
            <div class="gauge-bar"><div class="gauge-fill gf-amber" id="bar-inertia" style="width:0%"></div></div>
          </div>
        </div>
        <div class="mini-metric">
          <div class="mm-label">ENLACES ACOPL.</div>
          <div class="mm-val" style="color:var(--purple)" id="val-gates">—</div>
        </div>
      </div>

      <!-- Annealing / Exploration Temp -->
      <div class="metric-card">
        <div class="mc-title">
          TEMPERATURA DE EXPLORACIÓN
          <span class="mc-edu" onclick="showEdu('annealing')">?</span>
        </div>
        <div class="anneal-gauge">
          <div class="anneal-ring">
            <svg width="52" height="52" viewBox="0 0 52 52">
              <circle class="anneal-ring-bg" cx="26" cy="26" r="22"/>
              <circle class="anneal-ring-fill" id="anneal-circle" cx="26" cy="26" r="22"
                stroke="var(--amber)"
                stroke-dasharray="138.2"
                stroke-dashoffset="138.2"/>
            </svg>
            <div class="anneal-ring-val" id="anneal-val">—</div>
          </div>
          <div class="anneal-info">
            <b id="anneal-phase">—</b>
            Temperatura de exploración.<br>
            Alta → creativo · Baja → preciso
          </div>
        </div>
      </div>

      <!-- Collapse type -->
      <div class="collapse-display" id="collapse-display">
        <div class="mc-title">
          MODALIDAD DE DIRECTIVA GENERATIVA
          <span class="mc-edu" onclick="showEdu('collapse')">?</span>
        </div>
        <div class="cd-type" id="collapse-type" style="color:var(--text4)">— sin turno —</div>
        <div class="cd-script" id="collapse-script" style="display:none"></div>
      </div>

      <!-- Semantic weights -->
      <div class="metric-card">
        <div class="mc-title">PESOS SEMÁNTICOS ACTIVOS</div>
        <div class="sem-chart" id="sem-chart"></div>
      </div>

    </div>
  </div>

  <!-- ═══ CENTER: Chat ═══ -->
  <div class="chat-panel">
    <div class="panel-header">
      <span class="ph-icon">◈</span>
      INTERFACE CONVERSACIONAL
      <span style="margin-left:auto; font-family:var(--mono); font-size:0.55rem; color:var(--text4)" id="turn-counter">T#0</span>
    </div>

    <div class="chat-messages" id="chat-messages">
      <div style="text-align:center; padding: 40px 20px; color: var(--text4)">
        <div style="font-family:var(--display); font-size:0.75rem; letter-spacing:0.2em; color:var(--cyan); margin-bottom:12px">CHATBOT HÍBRIDO NEURO-SIMBÓLICO v8.1</div>
        <div style="font-family:var(--mono); font-size:0.65rem; line-height:1.8">
          Motor de contexto semántico 12D + TF-IDF/Logistic Regression + SLM local.<br>
          Operación 100% clásica sobre CPU con telemetría Green AI en tiempo real.
        </div>
        <div style="margin-top:20px; display:flex; justify-content:center; gap:8px; flex-wrap:wrap">
          <span class="tag tag-c">12 dimensiones semánticas (12D)</span>
          <span class="tag tag-p">TF-IDF + LR</span>
          <span class="tag tag-g">Continuidad Coseno</span>
          <span class="tag tag-a">Green AI Enabled</span>
        </div>
      </div>
    </div>

    <div class="chat-input-area">
      <textarea class="chat-input" id="chat-input"
        placeholder="Escribe tu mensaje… (Enter para enviar, Shift+Enter para nueva línea)"
        rows="1"></textarea>
      <button class="send-btn" id="send-btn" disabled onclick="sendMessage()">ENVIAR</button>
    </div>
  </div>

  <!-- ═══ RIGHT: System State ═══ -->
  <div class="panel">
    <div class="panel-header">
      <span class="ph-icon">◉</span> ESTADO DEL SISTEMA
    </div>
    <div class="panel-content">

      <!-- Dimension map (12D) -->
      <div class="edu-section">
        <div class="edu-title">⊞ Mapa de Dimensiones Semánticas (12D)</div>
        <div class="qubit-grid" id="qubit-grid">
          <!-- JS -->
        </div>
      </div>

      <!-- Semantic states / Dominant dimensions -->
      <div class="edu-section">
        <div class="edu-title">⊛ Dimensiones Dominantes</div>
        <div id="mv-states">
          <div style="font-family:var(--mono); font-size:0.6rem; color:var(--text4)">— esperando primer turno —</div>
        </div>
      </div>

      <!-- Coupling pairs -->
      <div class="edu-section">
        <div class="edu-title">↔ Pares Acoplados</div>
        <div id="ent-pairs">
          <div style="font-family:var(--mono); font-size:0.6rem; color:var(--text4)">— sin acoplamiento activo —</div>
        </div>
      </div>

      <!-- ML Classifier -->
      <div class="edu-section">
        <div class="edu-title">◈ ML Classifier (6 clases)</div>
        <div class="ml-bar-wrap" id="ml-bars">
          <!-- JS -->
        </div>
        <div style="font-family:var(--mono); font-size:0.58rem; color:var(--text4); margin-top:6px" id="ml-verdict">
          — esperando turno —
        </div>
      </div>

      <!-- Turn history -->
      <div class="edu-section">
        <div class="edu-title">◎ Historial de Turnos</div>
        <div class="turn-history" id="turn-history">
          <div style="font-family:var(--mono); font-size:0.6rem; color:var(--text4)">— sin turnos —</div>
        </div>
      </div>

      <!-- Config Hyperparameters -->
      <div class="edu-section">
        <div class="edu-title">⚙ Hiperparámetros Motor v8.1</div>
        <div style="font-family:var(--mono); font-size:0.62rem; color:var(--text2); display:grid; grid-template-columns:1fr 1fr; gap:6px;">
          <div>α (EWMA): <b style="color:var(--cyan)" id="val-alpha">0.30</b></div>
          <div>c (Acoplamiento): <b style="color:var(--purple)" id="val-coupling">0.50</b></div>
          <div>θ (Quiebre): <b style="color:var(--amber)" id="val-theta">0.50</b></div>
          <div>θ (Bajo/Alto): <b style="color:var(--text)">0.66 / 0.81</b></div>
          <div style="grid-column: span 2">SLM Model: <b style="color:var(--green)" id="val-slm-model">qwen2.5:1.5b</b></div>
        </div>
      </div>

      <!-- Config -->
      <div class="edu-section">
        <div class="edu-title">⚙ Control</div>
        <div class="ctrl-row">
          <div class="ctrl-label">LLM</div>
          <div class="ctrl-val" id="ctrl-llm-val">OFF</div>
          <button class="ctrl-btn" id="ctrl-llm-btn" onclick="toggleLLM()">ACTIVAR</button>
        </div>
        <div class="ctrl-row">
          <div class="ctrl-label">MODO</div>
          <div class="ctrl-val" id="ctrl-mode-val">rapido</div>
          <button class="ctrl-btn" onclick="toggleMode()">CAMBIAR</button>
        </div>
        <div class="ctrl-row">
          <div class="ctrl-label">RESET</div>
          <div class="ctrl-val" style="color:var(--text4); font-size:0.6rem;">estado semántico</div>
          <button class="ctrl-btn" onclick="resetQuantum()" style="border-color:var(--red); color:var(--red)">RESET</button>
        </div>
        <div class="ctrl-row">
          <div class="ctrl-label">ML RETRAIN</div>
          <div class="ctrl-val" style="color:var(--text4); font-size:0.6rem;">corpus actual</div>
          <button class="ctrl-btn" onclick="mlRetrain()">RETRAIN</button>
        </div>
      </div>

      <!-- Nota metodológica institucional UIDE -->
      <div style="margin-top:14px; padding:8px 10px; background:rgba(0,245,228,0.03); border:1px solid rgba(0,245,228,0.18); border-radius:4px; font-family:var(--mono); font-size:0.56rem; color:var(--text3); line-height:1.45;">
        🔬 <b style="color:var(--cyan)">Nota metodológica (UIDE):</b> El término <i>Quantum-Inspired</i> refiere a un marco conceptual basado en representaciones vectoriales en ℝ¹² ejecutadas estrictamente mediante álgebra lineal clásica y NumPy sobre CPU. No utiliza hardware cuántico real.
      </div>

    </div>
  </div>
</div>

<!-- Tooltips -->
<div class="tooltip-overlay" id="edu-overlay" onclick="closeEdu()">
  <div class="tooltip-box" onclick="event.stopPropagation()">
    <div class="tt-title" id="edu-title"></div>
    <div class="tt-body" id="edu-body"></div>
    <div class="tt-close"><button class="ctrl-btn" onclick="closeEdu()">CERRAR</button></div>
  </div>
</div>

<div class="tooltip-overlay" id="qubit-overlay" onclick="closeQubit()">
  <div class="tooltip-box" onclick="event.stopPropagation()">
    <div class="tt-title" id="qubit-title"></div>
    <div class="tt-body" id="qubit-body"></div>
    <div class="tt-close"><button class="ctrl-btn" onclick="closeQubit()">CERRAR</button></div>
  </div>
</div>

<script>
let apiBase = window.location.origin;
document.getElementById('apiUrl').value = apiBase;

let connected = false;
let currentMode = 'rapido';
let turnHistory = [];
let qubitData = [];
let lastQS = null;

const ML_LABELS = ['tecnologia','trabajo','estudio','bienestar','conversacional','general'];
const COLLAPSE_COLORS = {
  colapso_directo: 'var(--cyan)', superposicion_alta: 'var(--purple)',
  ruptura_semantica: 'var(--red)', decoherencia: 'var(--amber)',
  entropy_flush: 'var(--cyan2)', acoplamiento_detectado: 'var(--purple)', entrelazamiento_detectado: 'var(--purple)',
  interferencia_constructiva: 'var(--green)', interferencia_destructiva: 'var(--red)',
  colapso_martillo: 'var(--red)', primer_turno: 'var(--green)',
  convergencia_annealing: 'var(--amber)', conversacional: 'var(--text3)'
};
const SEM_COLORS = [
  'var(--cyan)', 'var(--purple)', 'var(--green)', 'var(--amber)',
  'var(--cyan2)', '#ff6eb4', 'var(--green)', 'var(--cyan)',
  'var(--purple)', 'var(--amber)', 'var(--red)', 'var(--green)'
];

async function api(path, method='GET', body=null) {
  const url = document.getElementById('apiUrl').value.replace(/\/$/, '') + path;
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function checkConnection() {
  try {
    const s = await api('/status');
    connected = true;
    document.getElementById('conn-dot').className = 'dot dot-on';
    document.getElementById('conn-label').textContent = 'CONECTADO';
    document.getElementById('send-btn').disabled = false;
    updateHeaderTags(s);
    await loadQubitMap();
    initMLBars();
    return true;
  } catch (e) {
    connected = false;
    document.getElementById('conn-dot').className = 'dot dot-off';
    document.getElementById('conn-label').textContent = 'DESCONECTADO';
    document.getElementById('send-btn').disabled = true;
    return false;
  }
}

function updateHeaderTags(s) {
  document.getElementById('tag-mode').textContent   = (s.mode || 'rapido').toUpperCase();
  document.getElementById('tag-qubits').textContent = (s.qubits || 12) + 'D';
  document.getElementById('tag-llm').textContent    = s.llm ? 'LLM ON' : 'LLM OFF';
  document.getElementById('tag-green').textContent  = '🍃 ' + (s.total_emissions_gco2 || 0).toFixed(3) + ' gCO₂';
  if (s.ollama_model) document.getElementById('val-slm-model').textContent = s.ollama_model;
  if (s.alpha) document.getElementById('val-alpha').textContent = s.alpha.toFixed(2);
  if (s.coupling) document.getElementById('val-coupling').textContent = s.coupling.toFixed(2);
  if (s.theta_quiebre) document.getElementById('val-theta').textContent = s.theta_quiebre.toFixed(2);
  document.getElementById('ctrl-llm-val').textContent = s.llm ? 'ON' : 'OFF';
  document.getElementById('ctrl-llm-btn').textContent = s.llm ? 'DESACTIVAR' : 'ACTIVAR';
  document.getElementById('ctrl-llm-btn').className   = 'ctrl-btn ' + (s.llm ? 'ctrl-btn-on' : 'ctrl-btn-off');
  document.getElementById('ctrl-mode-val').textContent = s.mode || 'rapido';
  currentMode = s.mode || 'rapido';
}

async function loadQubitMap() {
  try {
    const d = await api('/qubit-map');
    qubitData = d.qubits;
    renderQubitGrid(d.qubits);
  } catch {}
}

async function sendMessage() {
  const inp = document.getElementById('chat-input');
  const msg = inp.value.trim();
  if (!msg || !connected) return;

  inp.value = '';
  inp.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  appendMessage('user', msg);
  const thinkId = appendThinking();

  try {
    const res = await api('/chat', 'POST', { message: msg });
    removeThinking(thinkId);
    lastQS = res.quantum_signal;

    appendBotMessage(res.response, res.quantum_signal, res.source, res.total_ms);

    updateMetrics(res.quantum_signal);
    updateQubitMap(res.quantum_signal);
    updateMultiverse(res.quantum_signal);
    updateEntanglement(res.quantum_signal);
    updateMLBars(res.quantum_signal);
    addTurnHistory(res.quantum_signal, res.source);

    document.getElementById('turn-counter').textContent = 'T#' + res.quantum_signal.turn_number;
  } catch (e) {
    removeThinking(thinkId);
    appendError('Error de API: ' + e.message);
  }

  document.getElementById('send-btn').disabled = false;
  inp.focus();
}

function appendMessage(role, text) {
  const div = document.createElement('div');
  div.className = 'msg msg-' + role;
  const initials = role === 'user' ? 'YO' : '12D';
  const avatarClass = role === 'user' ? 'avatar-user' : 'avatar-bot';
  div.innerHTML = `
    <div class="msg-avatar ${avatarClass}">${initials}</div>
    <div class="msg-bubble bubble-${role}">${escapeHtml(text)}</div>
  `;
  const chat = document.getElementById('chat-messages');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function appendBotMessage(text, qs, source, ms) {
  const chat = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = 'msg';
  const ct   = qs.collapse_type || 'colapso_directo';
  const cc   = COLLAPSE_COLORS[ct] || 'var(--cyan)';
  const ml   = qs.ml_label || 'general';
  const conf = Math.round((qs.ml_confidence || 0) * 100);

  let formatted = escapeHtml(text)
    .replace(/```(\w*)\n?([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong style="color:var(--cyan2)">$1</strong>')
    .replace(/\n/g, '<br>');

  const co2 = (qs.emissions_gco2 !== undefined ? qs.emissions_gco2 : 0).toFixed(4);
  const watts = (qs.total_power_w !== undefined ? qs.total_power_w : 32.5).toFixed(1);

  const cleanScript = (qs.collapse_script || '⟨resolución directa⟩')
    .replace(/Circuito inicializado en superposición H\|0⟩/g, 'Estado semántico inicializado')
    .replace(/entrelazamiento/gi, 'acoplamiento')
    .replace(/Entrelazamiento/g, 'Acoplamiento')
    .replace(/colapso directo/gi, 'resolución directa')
    .replace(/superposicion alta/gi, 'exploración multidimensional')
    .replace(/colapso martillo/gi, 'resolución forzada');

  div.innerHTML = `
    <div class="msg-avatar avatar-bot">12D</div>
    <div>
      <div class="collapse-badge cb-${ct}" style="border-left-color:${cc}">
        ${cleanScript}
      </div>
      <div class="msg-bubble bubble-bot">${formatted}</div>
      <div class="msg-meta">
        <span><span class="meta-label">ML:</span> ${ml}(${conf}%)</span>
        <span><span class="meta-label">ent:</span> ${qs.entropy_vn !== undefined ? qs.entropy_vn : qs.entropy}</span>
        <span><span class="meta-label">fid:</span> ${qs.fidelity !== undefined ? qs.fidelity : qs.continuity}</span>
        <span><span class="meta-label">src:</span> ${source}</span>
        <span><span class="meta-label">ms:</span> ${ms}ms</span>
        <span style="color:var(--amber)"><span class="meta-label" style="color:var(--amber)">⚡</span> ${watts}W</span>
        <span style="color:var(--green)"><span class="meta-label" style="color:var(--green)">🍃</span> ${co2}gCO₂</span>
      </div>
    </div>
  `;

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function appendThinking() {
  const id  = 'think-' + Date.now();
  const div = document.createElement('div');
  div.id = id;
  div.className = 'thinking-indicator';
  div.innerHTML = `<span style="color:var(--text3)">PROCESANDO ESTADO SEMÁNTICO 12D</span><div class="thinking-dots"><span></span><span></span><span></span></div>`;
  const chat = document.getElementById('chat-messages');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendError(msg) {
  const div = document.createElement('div');
  div.style.cssText = 'font-family:var(--mono);font-size:0.65rem;color:var(--red);padding:8px;border:1px solid var(--red-dim);border-radius:3px';
  div.textContent = msg;
  document.getElementById('chat-messages').appendChild(div);
}

function updateMetrics(qs) {
  const ent = qs.entropy_vn !== undefined ? qs.entropy_vn : qs.entropy;
  const fid = qs.fidelity !== undefined ? qs.fidelity : qs.continuity;

  if (qs.emissions_gco2 !== undefined) {
    document.getElementById('val-co2-turn').textContent = qs.emissions_gco2.toFixed(4);
    document.getElementById('val-co2-total').textContent = (qs.total_emissions_gco2 || 0).toFixed(4);
    document.getElementById('tag-green').textContent = '🍃 ' + (qs.total_emissions_gco2 || 0).toFixed(3) + ' gCO₂';
    if (qs.total_power_w) document.getElementById('val-watts-total').textContent = qs.total_power_w.toFixed(1) + ' W';
    if (qs.cpu_power_w) document.getElementById('val-watts-cpu').textContent = qs.cpu_power_w.toFixed(1) + ' W';
    if (qs.ram_power_w) document.getElementById('val-watts-ram').textContent = qs.ram_power_w.toFixed(1) + ' W';
    if (qs.total_energy_kwh) document.getElementById('val-kwh-total').textContent = (qs.total_energy_kwh_acc || qs.total_energy_kwh).toFixed(6) + ' kWh';
    if (qs.water_consumed_l !== undefined) document.getElementById('val-water-l').textContent = (qs.water_consumed_l_acc || qs.water_consumed_l).toFixed(5) + ' L';
  }


  setGauge('entropy', ent, ent, `→ perspectivas: ${qs.n_perspectives || 2} · ${qs.structure_label || 'perspectiva dual'}`);
  setGauge('fidelity', fid, fid, `→ drift: ${qs.context_drift || 'continuo'}`);

  setMiniGauge('conc',    qs.concurrence, qs.concurrence);
  setMiniGauge('kl',      Math.min((qs.kl_divergence || 0) / 3, 1), qs.kl_divergence?.toFixed(4));
  setMiniGauge('inertia', qs.inertia, qs.inertia);
  document.getElementById('val-gates').textContent = qs.n_entangled_gates || 0;

  const a = qs.annealing_temp || 0.95;
  const circ = 138.2;
  const offset = circ * (1 - a);
  document.getElementById('anneal-circle').style.strokeDashoffset = offset;
  document.getElementById('anneal-val').textContent = a?.toFixed(2);
  document.getElementById('anneal-phase').textContent = a > 0.7 ? 'EXPLORACIÓN' : a > 0.4 ? 'TRANSICIÓN' : 'CONVERGENCIA';

  const ct = qs.collapse_type || 'colapso_directo';
  const cc = COLLAPSE_COLORS[ct] || 'var(--text3)';
  const ctDisplay = ct === 'entrelazamiento_detectado' ? 'ACOPLAMIENTO DETECTADO' :
                    ct === 'colapso_directo' ? 'RESOLUCIÓN DIRECTA' :
                    ct === 'colapso_martillo' ? 'RESOLUCIÓN FORZADA' :
                    ct === 'superposicion_alta' ? 'EXPLORACIÓN MULTIDIMENSIONAL' :
                    ct.replace(/_/g,' ')
                      .replace(/entrelazamiento/gi, 'acoplamiento')
                      .replace(/colapso/gi, 'resolución')
                      .toUpperCase();
  document.getElementById('collapse-type').textContent = ctDisplay;
  document.getElementById('collapse-type').style.color = cc;
  if (qs.collapse_script) {
    const cleanScript = qs.collapse_script
      .replace(/Circuito inicializado en superposición H\|0⟩/g, 'Estado semántico inicializado')
      .replace(/entrelazamiento/gi, 'acoplamiento')
      .replace(/Entrelazamiento/g, 'Acoplamiento')
      .replace(/colapso directo/gi, 'resolución directa')
      .replace(/superposicion alta/gi, 'exploración multidimensional')
      .replace(/colapso martillo/gi, 'resolución forzada');
    document.getElementById('collapse-script').textContent = cleanScript;
    document.getElementById('collapse-script').style.display = 'block';
    document.getElementById('collapse-script').style.borderLeftColor = cc;
  }

  const sw = qs.semantic_weights || {};
  const entries = Object.entries(sw).filter(([,v]) => v > 0).sort((a,b) => b[1]-a[1]).slice(0, 8);
  const chart = document.getElementById('sem-chart');
  chart.innerHTML = entries.map(([name, val], i) => {
    const pct = Math.round(val * 100);
    const color = SEM_COLORS[i % SEM_COLORS.length];
    return `<div class="sem-row">
      <div class="sem-name">${name}</div>
      <div class="sem-bar-bg">
        <div class="sem-bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
      <div class="sem-val">${val.toFixed(2)}</div>
    </div>`;
  }).join('') || '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text4)">— sin activación semántica —</div>';
}

function setGauge(id, val, raw, sub) {
  const pct = Math.round(Math.min(Math.max(val || 0, 0), 1) * 100);
  document.getElementById(`val-${id}`).textContent = (raw !== undefined ? raw : val)?.toFixed ? (raw?.toFixed(4) || '—') : '—';
  document.getElementById(`bar-${id}`).style.width = pct + '%';
  if (sub) document.getElementById(`sub-${id}`).textContent = sub;
}

function setMiniGauge(id, pctVal, displayVal) {
  const pct = Math.round(Math.min(Math.max(pctVal || 0, 0), 1) * 100);
  const el = document.getElementById(`val-${id}`);
  if (el) el.textContent = typeof displayVal === 'number' ? displayVal.toFixed(4) : (displayVal || '—');
  const bar = document.getElementById(`bar-${id}`);
  if (bar) bar.style.width = pct + '%';
}

function renderQubitGrid(qubits) {
  const grid = document.getElementById('qubit-grid');
  grid.innerHTML = qubits.slice(0, 12).map((q, i) => `
    <div class="qubit-cell ${q.active ? 'active' : ''}"
         onclick="showQubit(${i})" title="D${q.index}: ${q.name}">
      <div class="qc-index">D${q.index}</div>
      <div class="qc-name">${q.name.slice(0,4)}</div>
      <div class="qubit-weight-bar" style="width:${Math.round((q.weight||0)*100)}%"></div>
    </div>
  `).join('');
}

function updateQubitMap(qs) {
  if (!qubitData.length) return;
  const activeSet = new Set(qs.active_qubits || []);
  const sw = qs.semantic_weights || {};
  qubitData.forEach(q => {
    q.active = activeSet.has(q.index);
    q.weight = sw[q.name] || 0;
  });
  renderQubitGrid(qubitData);
}

function updateMultiverse(qs) {
  const states = qs.multiverse_states || [];
  const el = document.getElementById('mv-states');
  if (!states.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text4)">— sin datos —</div>';
    return;
  }
  el.innerHTML = states.map(([label, prob], i) => `
    <div class="mv-state mv-rank-${i}">
      <div class="mv-label">${label}</div>
      <div class="mv-bar-wrap"><div class="mv-bar" style="width:${Math.round(prob*100)}%"></div></div>
      <div class="mv-prob">${prob.toFixed(3)}</div>
    </div>
  `).join('');
}

function updateEntanglement(qs) {
  const pairs  = qs.entangled_names || [];
  const constr = new Set((qs.constructive_pairs || []).map(([a,b]) => a+'—'+b));
  const destr  = new Set((qs.destructive_pairs  || []).map(([a,b]) => a+'—'+b));
  const el = document.getElementById('ent-pairs');
  if (!pairs.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text4)">— sin acoplamiento activo —</div>';
    return;
  }
  el.innerHTML = pairs.slice(0, 5).map(([a, b]) => {
    const key = a+'—'+b;
    let typeClass = 'et-active', typeLabel = 'ACTIVO';
    if (constr.has(key)) { typeClass = 'et-constructiva'; typeLabel = 'CONSTR.'; }
    if (destr.has(key))  { typeClass = 'et-destructiva';  typeLabel = 'DESTR.'; }
    return `<div class="ent-pair">
      <span class="ent-a">${a}</span>
      <span class="ent-tensor"> — </span>
      <span class="ent-b">${b}</span>
      <span class="ent-type ${typeClass}">${typeLabel}</span>
    </div>`;
  }).join('');
}

function initMLBars() {
  const el = document.getElementById('ml-bars');
  el.innerHTML = ML_LABELS.map(lbl => `
    <div class="ml-row">
      <div class="ml-name">${lbl}</div>
      <div class="ml-bar-bg">
        <div class="ml-bar-fill" id="mlbar-${lbl}" style="width:0%"></div>
      </div>
      <div class="ml-conf" id="mlconf-${lbl}">—</div>
    </div>
  `).join('');
}

function updateMLBars(qs) {
  const label = qs.ml_label || 'general';
  const conf  = qs.ml_confidence || 0;
  ML_LABELS.forEach(lbl => {
    const bar  = document.getElementById('mlbar-' + lbl);
    const conf_el = document.getElementById('mlconf-' + lbl);
    if (!bar) return;
    if (lbl === label) {
      bar.style.width    = Math.round(conf * 100) + '%';
      bar.className      = 'ml-bar-fill ml-active';
      conf_el.textContent = Math.round(conf * 100) + '%';
      conf_el.style.color = 'var(--cyan)';
    } else {
      bar.style.width    = '0%';
      bar.className      = 'ml-bar-fill';
      conf_el.textContent = '—';
      conf_el.style.color = 'var(--text4)';
    }
  });
  document.getElementById('ml-verdict').textContent =
    `→ ${label.toUpperCase()} (${Math.round(conf*100)}% confianza)`;
}

function addTurnHistory(qs, source) {
  turnHistory.unshift({ qs, source });
  if (turnHistory.length > 8) turnHistory.pop();
  renderTurnHistory();
}

function renderTurnHistory() {
  const el = document.getElementById('turn-history');
  if (!turnHistory.length) {
    el.innerHTML = '<div style="font-family:var(--mono);font-size:0.6rem;color:var(--text4)">— sin turnos —</div>';
    return;
  }
  el.innerHTML = turnHistory.map(({ qs, source }) => {
    const ent = qs.entropy_vn !== undefined ? qs.entropy_vn : qs.entropy;
    const fid = qs.fidelity !== undefined ? qs.fidelity : qs.continuity;
    const rawCol = (qs.collapse_type || '').replace(/_/g, ' ');
    const colName = rawCol === 'entrelazamiento detectado' ? 'acoplamiento detectado' :
                    rawCol === 'colapso directo' ? 'resolución directa' :
                    rawCol === 'colapso martillo' ? 'resolución forzada' :
                    rawCol === 'superposicion alta' ? 'exploración multidimensional' :
                    rawCol.replace(/entrelazamiento/gi, 'acoplamiento')
                          .replace(/colapso/gi, 'resolución');
    return `
    <div class="turn-item">
      <span class="ti-num">#${qs.turn_number}</span>
      <span class="ti-ent">${ent?.toFixed(2)}</span>
      <span class="ti-fid">${fid?.toFixed(2)}</span>
      <span class="ti-col">${colName}</span>
      <span class="ti-source ts-${source}">${source.slice(0,3).toUpperCase()}</span>
    </div>
  `}).join('');
}

async function toggleLLM() {
  try {
    const status = await api('/status');
    const newState = !status.llm ? 'on' : 'off';
    const r = await api('/command', 'POST', { command: 'llm', value: newState });
    showToast(r.msg);
    const s = await api('/status');
    updateHeaderTags(s);
  } catch (e) { showToast('Error: ' + e.message); }
}

async function toggleMode() {
  const newMode = currentMode === 'rapido' ? 'profundo' : 'rapido';
  try {
    const r = await api('/command', 'POST', { command: 'modo', value: newMode });
    showToast(r.msg);
    currentMode = newMode;
    document.getElementById('ctrl-mode-val').textContent = newMode;
    document.getElementById('tag-mode').textContent = newMode.toUpperCase();
  } catch (e) { showToast('Error: ' + e.message); }
}

async function resetQuantum() {
  try {
    const r = await api('/command', 'POST', { command: 'reset' });
    showToast(r.msg);
    turnHistory = [];
    renderTurnHistory();
    document.getElementById('turn-counter').textContent = 'T#0';
    initMLBars();
  } catch (e) { showToast('Error: ' + e.message); }
}

async function mlRetrain() {
  try {
    const r = await api('/command', 'POST', { command: 'ml_retrain' });
    showToast(r.msg);
  } catch (e) { showToast('Error: ' + e.message); }
}

const EDU_CONTENT = {
  entropy: {
    title: 'ENTROPÍA SEMÁNTICA (Dispersión en ℝ¹²)',
    body: `Mide el grado de dispersión o focalización del vector semántico en el espacio de 12 dimensiones. Se calcula mediante una analogía matemática con la entropía de Shannon normalizada sobre probabilidades clásicas:
    <div class="tt-formula">H(p) = −∑ p_i log₂(p_i) / log₂(12)</div>
    • <b>H ≤ 0.66</b> → Directiva enfocada (dominancia de una sola dimensión)<br>
    • <b>0.66 < H < 0.81</b> → Directiva dual equilibrada<br>
    • <b>H ≥ 0.81</b> → Directiva exploratoria multidimensional`
  },
  fidelity: {
    title: 'FIDELIDAD SEMÁNTICA (Continuidad Coseno)',
    body: `Mide la similitud coseno entre el vector semántico del turno actual y el estado consolidado previo, regulado por el factor EWMA (α):
    <div class="tt-formula">C_t = ψ_t · ψ_{t-1} = cos(θ_{t, t-1})</div>
    Permite detectar quiebres temáticos cuando la continuidad cae por debajo de θ = 0.50.`
  },
  annealing: {
    title: 'TEMPERATURA DE EXPLORACIÓN',
    body: `Modula la amplitud de exploración directiva a lo largo de la conversación mediante una curva de decaimiento exponencial clásica:
    <div class="tt-formula">T(t) = 0.95 · exp(−0.07 · (t − 1))</div>
    Facilita mayor flexibilidad inicial y mayor precisión convergente en turnos avanzados.`
  },
  collapse: {
    title: 'MODALIDAD DE DIRECTIVA GENERATIVA',
    body: `Selecciona el marco de instrucción y enfoque simbólico que condiciona la inferencia del modelo SLM local en función del vector resultante en ℝ¹² y la matriz de acoplamiento C.`
  },
  green_ai: {
    title: 'GREEN AI — TELEMETRÍA CODECARBON (CPU/RAM)',
    body: `CodeCarbon estima en tiempo real la potencia instantánea (Watts), energía acumulada (kWh) y emisiones de CO₂ asociadas a la ejecución local sobre CPU y RAM.
    <div class="tt-formula">Emisiones (gCO₂eq) = Energía (kWh) × 0.1616 kgCO₂eq/kWh × 1000</div>
    • <b>Hardware monitoreado:</b> CPU x86-64 y memoria RAM (estimación de modelo de potencia).<br>
    • <b>Factor de emisión:</b> 0.1616 kgCO₂eq/kWh (Matriz eléctrica de Ecuador).<br>
    • <i>Nota metodológica:</i> Corresponde a telemetría de software y no a medición física directa en la toma eléctrica.`
  }
};

function showEdu(key) {
  const d = EDU_CONTENT[key];
  if (!d) return;
  document.getElementById('edu-title').textContent = d.title;
  document.getElementById('edu-body').innerHTML    = d.body;
  document.getElementById('edu-overlay').classList.add('show');
}
function closeEdu() { document.getElementById('edu-overlay').classList.remove('show'); }

function showQubit(idx) {
  const q = qubitData[idx];
  if (!q) return;
  document.getElementById('qubit-title').textContent = `D${q.index} — ${q.name.toUpperCase()} (Dimensión Semántica)`;
  document.getElementById('qubit-body').innerHTML =
    `<b style="color:var(--cyan)">${q.description}</b><br><br>
     Dimensión: <b style="color:var(--text)">D${q.index}</b><br>
     Estado: <span style="color:${q.active ? 'var(--green)' : 'var(--text4)'}">${q.active ? '▶ ACTIVA' : '◼ INACTIVA'}</span><br>
     Peso semántico normalizado: <span style="color:var(--amber)">${((q.weight||0) * 100).toFixed(1)}%</span><br><br>
     <b style="color:var(--text3)">Vocabulario clave asociado:</b><br>
     <div class="edu-tag-grid">${(q.keywords||[]).map(k => `<span class="edu-tag">${k}</span>`).join('')}</div>`;
  document.getElementById('qubit-overlay').classList.add('show');
}
function closeQubit() { document.getElementById('qubit-overlay').classList.remove('show'); }

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

let _toastTimer = null;
function showToast(msg) {
  const el = document.getElementById('collapse-script');
  el.textContent = '⚡ ' + msg;
  el.style.display = 'block';
  el.style.borderLeftColor = 'var(--amber)';
  el.style.color = 'var(--amber)';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.style.color = 'var(--text2)'; }, 3000);
}

document.getElementById('chat-input').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!document.getElementById('send-btn').disabled) sendMessage();
  }
});
document.getElementById('chat-input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 100) + 'px';
});

initMLBars();
checkConnection();
setInterval(() => { if (!connected) checkConnection(); }, 5000);
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SERVIDOR FASTAPI Y ENDPOINTS DEL LAB DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
bot_instance: Optional[HybridChatbot] = None

def get_bot_instance() -> HybridChatbot:
    global bot_instance
    if bot_instance is None:
        bot_instance = HybridChatbot(llm_enabled=True)
    return bot_instance

if _API_AVAILABLE:
    app = FastAPI(title="Quantum Hybrid Chatbot Lab", version=VERSION)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    class ChatRequest(BaseModel):
        message: str
        mode: Optional[str] = "rapido"

    class CommandRequest(BaseModel):
        command: str
        value: Optional[str] = None

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard():
        return DASHBOARD_HTML

    @app.get("/status")
    def get_status():
        bot = get_bot_instance()
        return {
            "mode": bot.mode,
            "qubits": N_DIMS,
            "llm": bot.llm_enabled,
            "ollama_available": bot.ollama_available,
            "ollama_model": bot.ollama_model,
            "turn": bot.engine.turn,
            "version": VERSION,
            "alpha": bot.engine.alpha,
            "coupling": bot.engine.coupling,
            "theta_quiebre": bot.engine.theta_quiebre,
            "total_emissions_gco2": round(bot.total_emissions_gco2, 4),
            "total_energy_kwh_acc": round(bot.total_energy_kwh_acc, 6),
            "total_water_l_acc": round(bot.total_water_l_acc, 5),
            "codecarbon_active": _CC_AVAILABLE
        }

    @app.get("/qubit-map")
    def get_qubit_map():
        bot = get_bot_instance()
        qubits = []
        last_qs = bot.last_signal
        active_set = set(last_qs.active_qubits) if last_qs else set()
        sw = last_qs.semantic_weights if last_qs else {}

        for i, (name, desc, keywords) in DIM_SEMANTICS.items():
            qubits.append({
                "index": i,
                "name": name,
                "description": desc,
                "keywords": keywords[:8],
                "active": i in active_set,
                "weight": sw.get(name, 0.0)
            })
        return {"qubits": qubits, "n_qubits": N_DIMS}

    @app.post("/chat")
    def chat_endpoint(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(400, "Mensaje vacío.")
        bot = get_bot_instance()
        res = bot.respond(req.message)
        return {
            "response": res["response"],
            "source": res["source"],
            "total_ms": res["total_ms"],
            "emissions_gco2": res["emissions_gco2"],
            "quantum_signal": res["quantum_signal"]
        }

    @app.post("/command")
    def command_endpoint(req: CommandRequest):
        cmd = req.command.lower().strip()
        val = (req.value or "").lower().strip()
        bot = get_bot_instance()

        if cmd == "reset":
            bot.reset()
            return {"ok": True, "msg": "Contexto semántico reiniciado."}
        if cmd == "llm":
            bot.llm_enabled = (val == "on" or val == "true")
            bot._check_ollama()
            state = "activado" if bot.llm_enabled else "desactivado"
            return {"ok": True, "msg": f"LLM local {state}."}
        if cmd == "modo":
            bot.mode = val if val in ["rapido", "profundo"] else "rapido"
            return {"ok": True, "msg": f"Modo de procesamiento → {bot.mode}"}
        if cmd == "ml_retrain":
            msg = bot.classifier.train()
            return {"ok": True, "msg": msg}

        raise HTTPException(400, f"Comando no reconocido: {cmd}")


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
import socket

def find_free_port(preferred_port=8080):
    for p in [preferred_port, 8000, 8088, 8050, 8888, 5000]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('127.0.0.1', p)) != 0:
                    return p
        except Exception:
            pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]

def main():
    if "--sensibilidad" in sys.argv or "--h3" in sys.argv:
        from sensibilidad_h3_paralelo import ejecutar_sensibilidad_h3
        ejecutar_sensibilidad_h3()
        return

    bot = get_bot_instance()
    print(BANNER)
    print(f"  Motor      : {N_DIMS} Dimensiones Semánticas (12D) | α={ALPHA_DEFAULT} | c={COUPLING_DEFAULT}")
    print(f"  Ollama SLM : {'ON' if bot.ollama_available else 'OFF (Modo Clásico Activo)'} ({bot.ollama_model})")
    print(f"  Green AI   : {'CodeCarbon Activado' if _CC_AVAILABLE else 'CodeCarbon NO Instalado'}")
    print(f"  Versión    : v{VERSION}")

    host = "127.0.0.1"
    port = find_free_port(8080)
    url = f"http://{host}:{port}"

    print(f"\n  🚀 Iniciando Lab Dashboard en: {url}\n")

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    if _API_AVAILABLE:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    else:
        print("Error: fastapi y uvicorn son requeridos para correr el Dashboard.")

if __name__ == "__main__":
    main()
