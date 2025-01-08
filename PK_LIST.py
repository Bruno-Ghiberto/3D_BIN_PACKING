import os
import re
import pandas as pd
import numpy as np
from py3dbp import Packer, Bin, Item
from py3dbp.constants import RotationType
import plotly.graph_objects as go

# ======================================
# 1. FUNCIÓN PARA DETECTAR ENGRANAJES
# ======================================
def es_engranaje(codigo_producto):
    """
    Determina si el producto corresponde a 'engranajes'
    según el patrón '1.XXX1.XXXXXXX'
    """
    # Patrón de ejemplo: ^1\.    -> empieza con "1."
    #                    \d{3}   -> luego 3 dígitos
    #                    1       -> luego "1"
    #                    \.      -> punto
    #                    \d{7}$  -> 7 dígitos hasta final
    patron = r'^1\.\d{3}1\.\d{7}$'
    return bool(re.match(patron, codigo_producto.strip()))


# ======================================
# 2. FUNCIONES AUXILIARES PARA CREAR BINS Y EMPACAR
# ======================================
def crear_bins(nombre_bin, width, height, depth, cantidad, max_weight=999999):
    """
    Crea 'cantidad' contenedores (Bin) con el nombre especificado 
    y las dimensiones dadas, y los retorna en una lista.
    Simula tener 'cantidad' bins iguales disponibles.
    """
    bins = []
    for i in range(cantidad):
        b = Bin(
            name=nombre_bin,
            width=float(width),
            height=float(height),
            depth=float(depth),
            max_weight=max_weight
        )
        bins.append(b)
    return bins


def empacar_items_en_bins(items, bins, bigger_first=False):
    """
    Empaca la lista de 'items' en la lista de 'bins' usando py3dbp.
    Retorna:
      - packer (objeto Packer ya 'packeado')
      - unfit_names: set de nombres de ítems que no cupieron
    """
    packer = Packer()
    # Agregamos todos los bins
    for b in bins:
        packer.add_bin(b)
    # Agregamos los items
    for it in items:
        packer.add_item(it)
    
    # Empaque:
    packer.pack(
        bigger_first=bigger_first,     # si es False, respeta el orden de lista. 
        distribute_items=True,
        number_of_decimals=0          # evita redondeos
    )
    unfit_names = {it.name for it in packer.unfit_items}
    return packer, unfit_names


# ======================================
# 3. EJEMPLO DE LECTURA DE EXCEL
#    (Adjusta rutas y nombres de archivo/hojas según tu caso)
# ======================================
archivo_packing_list = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PACKING-LOGISTICS-py3dbp\PACKING LIST.xlsx'
archivo_dimensiones = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PACKING-LOGISTICS-py3dbp\DIMENSIONES CAJAS-NORMALIZADO.xlsx'
archivo_pesos = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PACKING-LOGISTICS-py3dbp\PESO_P.T.xlsx'

dim_cajas = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJAS')
caja_producto = pd.read_excel(archivo_dimensiones, sheet_name='CAJA-PRODUCTO')
tam_cajones = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJONES')
packing_list = pd.read_excel(archivo_packing_list, sheet_name='P.L.')
pesos_productos = pd.read_excel(archivo_pesos, sheet_name='PRODUCTO-PESO')

# Limpieza de columnas y normalización
for df in [dim_cajas, caja_producto, tam_cajones, packing_list, pesos_productos]:
    df.columns = df.columns.str.strip()

dim_cajas['DESCRIPCION'] = dim_cajas['DESCRIPCION'].astype(str).str.strip()

caja_cols = ['CAJA_OP_1', 'CAJA_OP_2', 'CAJA_OP_3']
cantidad_cols = ['CANTIDAD x CAJA_OP_1', 'CANTIDAD_x_CAJA_OP_2', 'CANTIDAD_x_CAJA_OP_3']
for col in caja_cols:
    if col in caja_producto.columns:
        caja_producto[col] = caja_producto[col].astype(str).str.strip()

pesos_productos['CODIGO'] = pesos_productos['CODIGO'].astype(str).str.strip()
pesos_productos['PESO(kg)'] = pd.to_numeric(pesos_productos['PESO(kg)'], errors='coerce')
packing_list['CODIGO'] = packing_list['CODIGO'].astype(str).str.strip()

# Diccionarios para acceso rápido
pesos_productos_dict = pesos_productos.set_index('CODIGO')['PESO(kg)'].to_dict()
dimensiones_cajas = dim_cajas.set_index('DESCRIPCION').to_dict('index')
dimensiones_cajones = tam_cajones.set_index('CODIGO').to_dict('index')

# ======================================
# 4. GENERACIÓN DE LA RELACIÓN PRODUCTO -> CAJAS POSIBLES
# ======================================
producto_a_cajas = {}
for _, row in caja_producto.iterrows():
    producto = str(row['CODIGO']).strip()
    opciones = []
    for caja_col, cant_col in zip(caja_cols, cantidad_cols):
        caja_desc = row.get(caja_col)
        cantidad_por_caja = row.get(cant_col)
        if pd.notna(caja_desc) and pd.notna(cantidad_por_caja):
            caja_desc = str(caja_desc).strip()
            opciones.append({
                'DESCRIPCION_CAJA': caja_desc,
                'CANTIDAD_X_CAJA': int(cantidad_por_caja)
            })
    if opciones:
        producto_a_cajas[producto] = opciones


def obtener_peso_producto(codigo_producto):
    """Obtiene el peso del producto o 1.0 si no existe."""
    peso = pesos_productos_dict.get(codigo_producto)
    if peso is not None and not pd.isnull(peso):
        return float(peso)
    else:
        return 1.0  # valor por defecto


# ======================================
# 5. CREAR EL DATAFRAME 'asignacion_cajas' (cada caja que se arma)
# ======================================
asignacion_cajas_list = []
item_num = 0

for _, producto_row in packing_list.iterrows():
    codigo_producto = str(producto_row['CODIGO']).strip()
    descripcion_producto = producto_row['DESCRIPCION']
    cantidad_producto = int(producto_row['CANTIDAD'])

    if codigo_producto in producto_a_cajas:
        opciones = producto_a_cajas[codigo_producto]
        # Si deseas, puedes usar el 'VOLUMEN_m3' proveniente de dim_cajas
        for op in opciones:
            desc_caja = op['DESCRIPCION_CAJA']
            if desc_caja in dimensiones_cajas:
                op['VOLUMEN_m3'] = dimensiones_cajas[desc_caja]['VOLUMEN_m3']
            else:
                op['VOLUMEN_m3'] = float('inf')
        
        # Ordenar primero por CANTIDAD_X_CAJA de mayor a menor
        opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
        remaining_quantity = cantidad_producto
        peso_producto = obtener_peso_producto(codigo_producto)

        # 5a. Llenar cajas completas
        for op in opciones:
            desc_caja = op['DESCRIPCION_CAJA']
            cant_x_caja = op['CANTIDAD_X_CAJA']
            num_full_boxes = remaining_quantity // cant_x_caja
            for _ in range(num_full_boxes):
                item_num += 1
                peso_paquete = peso_producto * cant_x_caja
                asignacion_cajas_list.append({
                    'ITEM': item_num,
                    'CAJA': desc_caja,
                    'CODIGO': codigo_producto,
                    'DESCRIPCION': descripcion_producto,
                    'CANTIDAD': cant_x_caja,
                    'PESO': peso_paquete,
                    'VOLUMEN': op['VOLUMEN_m3']
                })
            remaining_quantity -= num_full_boxes * cant_x_caja

        # 5b. Sobran unidades => buscar caja que las soporte
        if remaining_quantity > 0:
            # Ordenar por CANTIDAD_X_CAJA de menor a mayor
            opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'])
            caja_asignada = False
            for op in opciones:
                if op['CANTIDAD_X_CAJA'] >= remaining_quantity:
                    desc_caja = op['DESCRIPCION_CAJA']
                    item_num += 1
                    peso_paquete = peso_producto * remaining_quantity
                    asignacion_cajas_list.append({
                        'ITEM': item_num,
                        'CAJA': desc_caja,
                        'CODIGO': codigo_producto,
                        'DESCRIPCION': descripcion_producto,
                        'CANTIDAD': remaining_quantity,
                        'PESO': peso_paquete,
                        'VOLUMEN': op['VOLUMEN_m3']
                    })
                    remaining_quantity = 0
                    caja_asignada = True
                    break

            if not caja_asignada and remaining_quantity > 0:
                # Si ninguna menor lo soporta, usar la mayor
                opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
                op = opciones[0]
                desc_caja = op['DESCRIPCION_CAJA']
                item_num += 1
                peso_paquete = peso_producto * remaining_quantity
                asignacion_cajas_list.append({
                    'ITEM': item_num,
                    'CAJA': desc_caja,
                    'CODIGO': codigo_producto,
                    'DESCRIPCION': descripcion_producto,
                    'CANTIDAD': remaining_quantity,
                    'PESO': peso_paquete,
                    'VOLUMEN': op['VOLUMEN_m3']
                })
                remaining_quantity = 0
    else:
        print(f"No hay cajas definidas para el producto '{codigo_producto}'")

asignacion_cajas = pd.DataFrame(asignacion_cajas_list)


# ======================================
# 6. ORDEN POR PESO DESCENDENTE
#    (para que py3dbp intente primero lo más pesado)
# ======================================
asignacion_cajas.sort_values(by='PESO', ascending=False, inplace=True)

# ======================================
# 7. AÑADIMOS DIMENSIONES DE LA CAJA EMM
# ======================================
asignacion_cajas = asignacion_cajas.merge(
    dim_cajas[['DESCRIPCION', 'ANCHO_(mm)', 'ALTO_(mm)', 'LARGO_(mm)']],
    left_on='CAJA',
    right_on='DESCRIPCION',
    how='left',
    suffixes=('_prod', '_dim')
)

if 'DESCRIPCION_prod' in asignacion_cajas.columns:
    asignacion_cajas.rename(columns={'DESCRIPCION_prod': 'DESCRIPCION'}, inplace=True)
if 'DESCRIPCION_dim' in asignacion_cajas.columns:
    asignacion_cajas.drop(columns=['DESCRIPCION_dim'], inplace=True, errors='ignore')


# ======================================
# 8. SEPARAMOS ÍTEMS: ENGRANAJES Y OTROS
# ======================================
# Creamos "Item" de py3dbp
all_items = []
for idx, row in asignacion_cajas.iterrows():
    it = Item(
        name=f"ITEM_{int(row['ITEM'])}",
        width=float(row['LARGO_(mm)']),
        height=float(row['ANCHO_(mm)']),
        depth=float(row['ALTO_(mm)']),
        weight=float(row['PESO'])
    )
    it.rotation_type = RotationType.ALL
    all_items.append(it)

engranajes_items = []
otros_items = []

for it, row in zip(all_items, asignacion_cajas.itertuples()):
    codigo_prod = getattr(row, 'CODIGO')
    if es_engranaje(codigo_prod):
        engranajes_items.append(it)
    else:
        otros_items.append(it)


# ======================================
# 9. CAJONES “ILIMITADOS” => CREACIÓN DE MUCHOS BINS
#    FASE 1: Exclusivos para engranajes
#    FASE 2: Mixtos (para todo lo demás y sobrantes de engranajes)
# ======================================

# Por ejemplo, definimos 200 bins EXCLUSIVOS para engranajes
bins_engranajes = crear_bins(
    nombre_bin="CAJON_EXCLUSIVO_ENGRANAJES",
    width=890,
    height=860,
    depth=1040,
    cantidad=200
)

# Definimos 300 bins MIXTOS
bins_mixtos = crear_bins(
    nombre_bin="CAJON_MIXTO",
    width=890,
    height=860,
    depth=1040,
    cantidad=300
)

# (Si quieres también el cajón partido, podrías crear 150 bins partidos y
#  agregarlos a bins_mixtos, por ejemplo.)

# ========== FASE 1: Empacar engranajes ==========
packer_eng, unfit_eng = empacar_items_en_bins(engranajes_items, bins_engranajes, bigger_first=False)

# Los sobrantes de engranajes van a la fase 2
sobrantes_engranajes = list(packer_eng.unfit_items)

# ========== FASE 2: Empacar “otros” + sobrantes de engranajes ==========
items_fase2 = otros_items + sobrantes_engranajes
packer_mix, unfit_mix = empacar_items_en_bins(items_fase2, bins_mixtos, bigger_first=False)

# Items sin asignar en total
unfit_names_total = {it.name for it in packer_eng.unfit_items} | {it.name for it in packer_mix.unfit_items}
print("Ítems sin asignar:", unfit_names_total if unfit_names_total else "Ninguno")


# ======================================
# 10. MAPEO DE ÍTEM -> CAJÓN USADO
# ======================================
item_to_cajon = {}
cajon_counter = 0

# Fase 1
for b in packer_eng.bins:
    if b.items:
        cajon_counter += 1
        for it in b.items:
            item_to_cajon[it.name] = (b.name, cajon_counter, it.position, it.width, it.height, it.depth)

# Fase 2
for b in packer_mix.bins:
    if b.items:
        cajon_counter += 1
        for it in b.items:
            item_to_cajon[it.name] = (b.name, cajon_counter, it.position, it.width, it.height, it.depth)


def asignar_cajon(item_id):
    key = f"ITEM_{item_id}"
    if key in unfit_names_total:
        return ("SIN ASIGNAR", None)
    elif key in item_to_cajon:
        return (item_to_cajon[key][0], item_to_cajon[key][1])
    else:
        return ("SIN ASIGNAR", None)

asignacion_cajas['DESCRIPCION_DE_CAJON'] = asignacion_cajas['ITEM'].apply(lambda x: asignar_cajon(x)[0])
asignacion_cajas['#_DE_CAJON'] = asignacion_cajas['ITEM'].apply(lambda x: asignar_cajon(x)[1])

# Exportar a Excel
asignacion_cajas.to_excel("asignacion_cajas_final.xlsx", index=False)
print("Archivo exportado: asignacion_cajas_final.xlsx")


# ======================================
# 11. GENERAR VISUALIZACIONES 3D (OPCIONAL)
# ======================================
if not os.path.exists('cajones_3d'):
    os.makedirs('cajones_3d')

colors = ['blue', 'red', 'green', 'orange', 'purple', 'yellow', 
          'cyan', 'magenta', 'lime', 'pink', 'gold', 'teal']

# Caras de un cubo
i_faces = [0,0,4,4,0,0,1,1,0,0,3,3]
j_faces = [1,2,5,6,3,7,2,6,1,5,2,6]
k_faces = [2,3,6,7,7,4,6,5,5,4,6,7]

def graficar_packer(packer_obj, nombre_fase):
    idx_bin = 0
    for b in packer_obj.bins:
        if not b.items:
            continue
        idx_bin += 1

        fig = go.Figure()

        w = b.width
        h = b.height
        d = b.depth

        # Coordenadas del contenedor
        X_cajon = [0,    w,    w,    0,    0,    w,    w,    0]
        Y_cajon = [0,    0,    h,    h,    0,    0,    h,    h]
        Z_cajon = [0,    0,    0,    0,    d,    d,    d,    d]

        fig.add_trace(go.Mesh3d(
            x=X_cajon,
            y=Y_cajon,
            z=Z_cajon,
            i=i_faces, j=j_faces, k=k_faces,
            color='lightgrey',
            opacity=0.4,
            name=f'{b.name}'
        ))

        color_index = 0
        for it in b.items:
            ix, iy, iz = it.position
            iw, ih, id_ = it.width, it.height, it.depth

            X = [ix, ix+iw, ix+iw, ix, ix, ix+iw, ix+iw, ix]
            Y = [iy, iy, iy+ih, iy+ih, iy, iy, iy+ih, iy+ih]
            Z = [iz, iz, iz, iz, iz+id_, iz+id_, iz+id_, iz+id_]

            # parseamos el item_id
            item_id = int(it.name.split('_')[1])
            row_item = asignacion_cajas[asignacion_cajas['ITEM'] == item_id].iloc[0]

            hover_text = (
                f"ITEM: {item_id}<br>"
                f"CODIGO: {row_item['CODIGO']}<br>"
                f"DESCRIPCION: {row_item['DESCRIPCION']}<br>"
                f"CANTIDAD: {row_item['CANTIDAD']}<br>"
                f"PESO: {row_item['PESO']:.2f} kg"
            )

            fig.add_trace(go.Mesh3d(
                x=X, y=Y, z=Z,
                i=i_faces, j=j_faces, k=k_faces,
                color=colors[color_index % len(colors)],
                opacity=1,
                name=f"Item {item_id}",
                hovertext=hover_text,
                hoverinfo='text'
            ))
            color_index += 1

        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm) - LARGO',
                yaxis_title='Y (mm) - ANCHO',
                zaxis_title='Z (mm) - ALTO'
            ),
            title=f'{nombre_fase} - Bin {idx_bin}: {b.name}'
        )
        fig.write_html(f'cajones_3d/{nombre_fase}_bin_{idx_bin}.html')

# Graficar resultados de Fase 1 (engranajes) y Fase 2 (mixtos)
graficar_packer(packer_eng, "FASE1_ENGRANAJES")
graficar_packer(packer_mix, "FASE2_MIXTO")

print("Visualizaciones 3D generadas en la carpeta 'cajones_3d'.")


# ======================================
# 12. VERIFICACIÓN FINAL DE SOBRESALIR
# ======================================
for b in packer_eng.bins:
    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth
        if ix + iw > b.width or iy + ih > b.height or iz + id_ > b.depth:
            print(f"[FASE1] El ítem {it.name} sobresale del cajón {b.name}")

for b in packer_mix.bins:
    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth
        if ix + iw > b.width or iy + ih > b.height or iz + id_ > b.depth:
            print(f"[FASE2] El ítem {it.name} sobresale del cajón {b.name}")

