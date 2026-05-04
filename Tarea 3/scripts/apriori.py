import pandas as pd
from itertools import combinations
from typing import List, Tuple, Dict


class Apriori:
    """
    Implementación del algoritmo Apriori con caché de cálculos.
    """

    def __init__(self, soporte_min: float = 0.5, confianza_min: float = 0.5):
        """
        Inicializa el algoritmo Apriori optimizado.

        Parámetros:
        -----------
        soporte_min : float
            Umbral de soporte mínimo (0.0 a 1.0). Default: 0.5
        confianza_min : float
            Umbral de confianza mínimo (0.0 a 1.0). Default: 0.5

        Regresa:
        --------
        None
        """
        self.soporte_min = soporte_min
        self.confianza_min = confianza_min
        self.transacciones = []
        self.itemsets_frecuentes = {}
        self.reglas_asociacion = []

        # Caché de soportes
        self.cache_soporte = {}

        # Indexar transacciones
        self.num_transacciones = 0

    def carga_transacciones(
        self, datos: pd.DataFrame, columnas: List[str] = None
    ) -> None:
        """
        Carga transacciones desde un DataFrame de pandas.

        Parámetros:
        -----------
        datos : pd.DataFrame
            DataFrame con los datos de transacciones
        columnas : List[str], optional
            Columnas a utilizar. Si es None, usa todas menos la primera (ID).

        Regresa:
        --------
        None
        """
        if columnas is None:
            columnas = datos.columns[1:]

        self.transacciones = []
        for idx, fila in datos.iterrows():
            transaccion = []
            for columna in columnas:
                if pd.notna(fila[columna]) and fila[columna] != "":
                    transaccion.append(f"{columna}={fila[columna]}")
            if transaccion:
                self.transacciones.append(frozenset(transaccion))

        # Guardar número de transacciones
        self.num_transacciones = len(self.transacciones)

        # Convertir a conjunto de conjuntos para búsqueda más rápida
        self.transacciones_set = [set(t) for t in self.transacciones]

        print(f"✓ {self.num_transacciones} transacciones cargadas")

    def calcula_soporte(self, itemset: frozenset) -> float:
        """
        Calcula el soporte de un itemset (con caché).

        Parámetros:
        -----------
        itemset : frozenset
            Conjunto de items para el cual calcular el soporte

        Regresa:
        --------
        float
            Valor de soporte entre 0.0 y 1.0
        """
        # Verificar caché primero
        if itemset in self.cache_soporte:
            return self.cache_soporte[itemset]

        # Contar transacciones que contienen el itemset
        contador = sum(
            1 for transaccion in self.transacciones_set if itemset.issubset(transaccion)
        )
        soporte = (
            contador / self.num_transacciones if self.num_transacciones > 0 else 0.0
        )

        # Guardar en caché
        self.cache_soporte[itemset] = soporte

        return soporte

    def generar_candidatos(self, itemsets: List[frozenset], k: int) -> List[frozenset]:
        """
        Genera candidatos de forma más eficiente (F-k-join).

        Parámetros:
        -----------
        itemsets : List[frozenset]
            Lista de itemsets frecuentes del nivel anterior
        k : int
            Tamaño del itemset a generar

        Regresa:
        --------
        List[frozenset]
            Lista de candidatos generados
        """
        if k == 2:
            # Para k=2, simplemente hacer combinaciones de items
            items = set()
            for itemset in itemsets:
                items.update(itemset)
            items = sorted(list(items))
            candidatos = [
                frozenset([items[i], items[j]])
                for i in range(len(items))
                for j in range(i + 1, len(items))
            ]
        else:
            # Para k>2, unir itemsets que comparten k-2 items
            candidatos = set()
            itemsets_sorted = sorted([sorted(list(x)) for x in itemsets])

            for i in range(len(itemsets_sorted)):
                for j in range(i + 1, len(itemsets_sorted)):
                    # Si comparten los primeros k-2 elementos
                    if itemsets_sorted[i][:-1] == itemsets_sorted[j][:-1]:
                        union = frozenset(itemsets_sorted[i]) | frozenset(
                            itemsets_sorted[j]
                        )
                        if len(union) == k:
                            candidatos.add(union)

            candidatos = list(candidatos)

        return candidatos

    def obtiene_itemsets_frecuentes(self, itemsets: List[frozenset]) -> List[frozenset]:
        """
        Filtra itemsets por el umbral de soporte mínimo.

        Parámetros:
        -----------
        itemsets : List[frozenset]
            Lista de itemsets candidatos

        Regresa:
        --------
        List[frozenset]
            Lista de itemsets que cumplen con soporte_min
        """
        frecuentes = []
        for itemset in itemsets:
            if self.calcula_soporte(itemset) >= self.soporte_min:
                frecuentes.append(itemset)

        return frecuentes

    def apriori(self) -> Dict[int, List[Tuple[frozenset, float]]]:
        """
        Ejecuta el algoritmo Apriori completo (optimizado).

        Parámetros:
        -----------
        (ninguno)

        Regresa:
        --------
        Dict[int, List[Tuple[frozenset, float]]]
            Diccionario de itemsets frecuentes por nivel
        """
        if not self.transacciones:
            raise ValueError(
                "No hay transacciones cargadas. Usa carga_transacciones() primero."
            )

        print(f"\n Ejecutando Apriori (soporte_min={self.soporte_min})...")

        self.itemsets_frecuentes = {}

        # Genera itemsets de tamaño 1
        print(" Generando itemsets de tamaño 1...")
        items = set()
        for transaccion in self.transacciones:
            items.update(transaccion)

        candidatos_1 = [frozenset([item]) for item in items]
        frecuentes_1 = self.obtiene_itemsets_frecuentes(candidatos_1)

        if not frecuentes_1:
            print(" ✗ No se encontraron itemsets frecuentes")
            return self.itemsets_frecuentes

        self.itemsets_frecuentes[1] = [
            (itemset, self.calcula_soporte(itemset)) for itemset in frecuentes_1
        ]

        print(f" ✓ {len(frecuentes_1)} itemsets de tamaño 1")

        # Genera itemsets de tamaño k > 1
        k = 2
        frecuentes_actuales = frecuentes_1

        while frecuentes_actuales:
            print(f" Generando itemsets de tamaño {k}...")

            candidatos_k = self.generar_candidatos(frecuentes_actuales, k)

            if not candidatos_k:
                break

            frecuentes_k = self.obtiene_itemsets_frecuentes(candidatos_k)

            if not frecuentes_k:
                print(f" ✓ {len(frecuentes_k)} itemsets de tamaño {k}")
                break

            self.itemsets_frecuentes[k] = [
                (itemset, self.calcula_soporte(itemset)) for itemset in frecuentes_k
            ]

            print(f" ✓ {len(frecuentes_k)} itemsets de tamaño {k}")

            frecuentes_actuales = frecuentes_k
            k += 1

        print(
            f"✓ Total de itemsets: {sum(len(v) for v in self.itemsets_frecuentes.values())}"
        )

        return self.itemsets_frecuentes

    def calcula_confianza(
        self, antecedente: frozenset, consecuente: frozenset
    ) -> float:
        """
        Calcula la confianza (con caché).

        Parámetros:
        -----------
        antecedente : frozenset
            Conjunto de items antecedentes
        consecuente : frozenset
            Conjunto de items consecuentes

        Regresa:
        --------
        float
            Valor de confianza entre 0.0 y 1.0
        """
        soporte_antecedente = self.calcula_soporte(antecedente)
        if soporte_antecedente == 0:
            return 0.0

        soporte_union = self.calcula_soporte(antecedente | consecuente)
        return soporte_union / soporte_antecedente

    def calcula_lift(self, antecedente: frozenset, consecuente: frozenset) -> float:
        """
        Calcula el lift (con caché).

        Parámetros:
        -----------
        antecedente : frozenset
            Conjunto de items antecedentes
        consecuente : frozenset
            Conjunto de items consecuentes

        Regresa:
        --------
        float
            Valor de lift
        """
        confianza = self.calcula_confianza(antecedente, consecuente)
        soporte_consecuente = self.calcula_soporte(consecuente)

        if soporte_consecuente == 0:
            return 0.0

        return confianza / soporte_consecuente

    def genera_reglas_asociacion(self) -> List[Dict]:
        """
        Genera reglas de asociación (optimizado).

        Parámetros:
        -----------
        (ninguno)

        Regresa:
        --------
        List[Dict]
            Lista de reglas de asociación
        """
        if not self.itemsets_frecuentes:
            raise ValueError("Primero debes ejecutar apriori()")

        print(f"\nGenerando reglas (confianza_min={self.confianza_min})...")

        self.reglas_asociacion = []
        contador_reglas = 0

        # Procesa itemsets de tamaño >= 2
        for k in sorted(self.itemsets_frecuentes.keys()):
            if k < 2:
                continue

            num_itemsets_k = len(self.itemsets_frecuentes[k])

            for idx, (itemset, soporte) in enumerate(self.itemsets_frecuentes[k]):
                if (idx + 1) % max(1, num_itemsets_k // 10) == 0:
                    print(f"  Procesando nivel {k}: {idx + 1}/{num_itemsets_k}")

                items = list(itemset)

                # Solo generar particiones, no todas las combinaciones
                for r in range(1, len(items)):
                    for items_antecedente in combinations(items, r):
                        antecedente = frozenset(items_antecedente)
                        consecuente = itemset - antecedente

                        confianza = self.calcula_confianza(antecedente, consecuente)

                        if confianza >= self.confianza_min:
                            lift = self.calcula_lift(antecedente, consecuente)

                            self.reglas_asociacion.append(
                                {
                                    "antecedente": antecedente,
                                    "consecuente": consecuente,
                                    "soporte": soporte,
                                    "confianza": confianza,
                                    "lift": lift,
                                }
                            )

                            contador_reglas += 1

        print(f"{contador_reglas} reglas generadas")

        return self.reglas_asociacion

    def obtiene_reglas_dataframe(self) -> pd.DataFrame:
        """
        Convierte las reglas generadas a un DataFrame de pandas.

        Parámetros:
        -----------
        (ninguno)

        Regresa:
        --------
        pd.DataFrame
            DataFrame con las reglas
        """
        if not self.reglas_asociacion:
            raise ValueError("Primero debes ejecutar genera_reglas_asociacion()")

        df = pd.DataFrame(
            {
                "antecedente": [
                    regla["antecedente"] for regla in self.reglas_asociacion
                ],
                "consecuente": [
                    regla["consecuente"] for regla in self.reglas_asociacion
                ],
                "soporte": [regla["soporte"] for regla in self.reglas_asociacion],
                "confianza": [regla["confianza"] for regla in self.reglas_asociacion],
                "lift": [regla["lift"] for regla in self.reglas_asociacion],
            }
        )

        return df.sort_values("lift", ascending=False).reset_index(drop=True)

    def obtiene_itemsets_frecuentes_dataframe(self) -> pd.DataFrame:
        """
        Convierte los itemsets frecuentes a un DataFrame de pandas.

        Parámetros:
        -----------
        (ninguno)

        Regresa:
        --------
        pd.DataFrame
            DataFrame con los itemsets
        """
        datos = []
        for k, lista_itemsets in sorted(self.itemsets_frecuentes.items()):
            for itemset, soporte in lista_itemsets:
                datos.append({"itemset": itemset, "soporte": soporte, "tamaño": k})

        df = pd.DataFrame(datos)
        return df.sort_values("soporte", ascending=False).reset_index(drop=True)
