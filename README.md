# 3D-BPP: Solucionador del Problema de Empaquetado en Contenedores 3D para Logística (PROTOTIPO EN DESARROLLO)

## Descripción

Este repositorio contiene una solución integral para el Problema de Empaquetado en Contenedores 3D (3D-BPP) diseñada específicamente para aplicaciones logísticas. El sistema organiza eficientemente cajas de diferentes dimensiones en contenedores o cajones, optimizando la utilización del espacio. Procesa datos de productos de sistemas logísticos, calcula ubicaciones óptimas y genera visualizaciones 3D interactivas de la disposición del empaquetado.

## Características Principales

- **Algoritmo de empaquetado 3D basado en heurísticas** que coloca eficientemente los artículos en contenedores
- **Pipeline de procesamiento de datos** que gestiona información de productos desde archivos Excel/CSV
- **Visualizaciones 3D interactivas** de las soluciones de empaquetado con elementos codificados por colores
- **Integración con información logística** incluyendo códigos de productos, descripciones y cantidades
- **Enfoque de empaquetado basado en estanterías** que simula procedimientos de carga realistas
- **Resultados exportables** en formato CSV y visualizaciones HTML interactivas

## Estructura del Repositorio

El repositorio consta de tres componentes principales:

### 1. Módulo Principal (`MAIN.py`)
El punto central de ejecución que coordina todo el proceso de empaquetado mediante:
- Carga de datos de cajas desde archivos CSV
- Configuración y ejecución del algoritmo de empaquetado
- Combinación de resultados con la información original del producto
- Generación de representaciones visuales de la solución

### 2. Módulo de Visualización (`Plotter.py`)
Un sistema especializado de visualización 3D que:
- Crea visualizaciones 3D interactivas utilizando Plotly
- Representa cada caja como un prisma semitransparente con bordes de estructura alámbrica
- Codifica las cajas por color según su tipo para facilitar la identificación
- Incluye información emergente para detalles de caja (ID, tipo, descripción, cantidad)
- Exporta visualizaciones como archivos HTML interactivos

### 3. Módulo de Utilidades (`Utils.py`)
Funciones de soporte que:
- Procesan listas de empaque desde archivos Excel
- Calculan asignaciones de producto a caja
- Determinan selecciones óptimas de cajas basadas en cantidades de productos
- Gestionan datos dimensionales y traducciones de coordenadas

## Uso

1. Prepare sus datos de productos en un archivo Excel con el formato adecuado
2. Configure las dimensiones del contenedor en el script principal
3. Ejecute el script principal para ejecutar el algoritmo de empaquetado:
   ```
   python MAIN.py
   ```
4. Revise el archivo CSV generado con las coordenadas de empaquetado
5. Abra los archivos de visualización HTML para inspeccionar la disposición del empaquetado 3D

## Dependencias

- pandas: Para manipulación de datos
- plotly: Para visualización 3D
- numpy: Para operaciones numéricas

## Resultados

El sistema genera dos resultados principales:
1. Un archivo CSV (`placements_result.csv`) que contiene información detallada de empaquetado incluyendo:
   - ID de caja, descripción y detalles de contenido
   - Coordenadas 3D para cada elemento colocado
   - Asignaciones de contenedor

2. Visualizaciones HTML interactivas para cada contenedor mostrando la disposición 3D de las cajas con codificación de colores e información emergente para facilitar la inspección de la solución de empaquetado.

Esta solución está diseñada para optimizar operaciones logísticas maximizando la utilización de contenedores mientras proporciona retroalimentación visual clara sobre la disposición del empaquetado.
