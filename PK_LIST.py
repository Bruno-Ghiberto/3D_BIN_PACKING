import pandas as pd
import numpy as np
from py3dbp import Packer, Bin, Item
from py3dbp.constants import RotationType
import plotly.graph_objects as go
import os

# ======================================
# CONFIGURACIÓN DE ARCHIVOS DE ENTRADA
# ======================================
archivo_packing_list = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PROGRAMA PYTHON\C.L-01\PACKING LIST.xlsx'
archivo_dimensiones = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PROGRAMA PYTHON\C.L-01\DIMENSIONES CAJAS-NORMALIZADO.xlsx'
archivo_pesos = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\PROGRAMA PYTHON\C.L-01\PESO_P.T.xlsx'

# ======================================
# CARGA DE DATOS DESDE EXCEL
# ======================================
dim_cajas = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJAS')
caja_producto = pd.read_excel(archivo_dimensiones, sheet_name='CAJA-PRODUCTO')
tam_cajones = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJONES')
packing_list = pd.read_excel(archivo_packing_list, sheet_name='P.L.')
pesos_productos = pd.read_excel(archivo_pesos, sheet_name='PRODUCTO-PESO')

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

pesos_productos_dict = pesos_productos.set_index('CODIGO')['PESO(kg)'].to_dict()
dimensiones_cajas = dim_cajas.set_index('DESCRIPCION').to_dict('index')
dimensiones_cajones = tam_cajones.set_index('CODIGO').to_dict('index')

# Mapeamos cada producto a sus posibles opciones de caja
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
    peso = pesos_productos_dict.get(codigo_producto)
    if peso is not None and not pd.isnull(peso):
        return float(peso)
    else:
        # Si no encontramos el producto, asignar un peso fijo
        return 1.0

# ======================================
# 1) CREAR EL DATAFRAME asignacion_cajas
# ======================================
asignacion_cajas_list = []
item_num = 0

for _, producto_row in packing_list.iterrows():
    codigo_producto = str(producto_row['CODIGO']).strip()
    descripcion_producto = producto_row['DESCRIPCION']
    cantidad_producto = int(producto_row['CANTIDAD'])

    if codigo_producto in producto_a_cajas:
        opciones = producto_a_cajas[codigo_producto]
        # Guardamos el volumen de cada caja
        for op in opciones:
            desc_caja = op['DESCRIPCION_CAJA']
            if desc_caja in dimensiones_cajas:
                op['VOLUMEN_m3'] = dimensiones_cajas[desc_caja]['VOLUMEN_m3']
            else:
                op['VOLUMEN_m3'] = float('inf')

        # Ordenamos las opciones de mayor a menor capacidad
        opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
        remaining_quantity = cantidad_producto
        peso_producto = obtener_peso_producto(codigo_producto)

        # Empaquetar en cajas completas
        for op in opciones:
            desc_caja = op['DESCRIPCION_CAJA']
            cant_x_caja = op['CANTIDAD_X_CAJA']
            num_full_boxes = remaining_quantity // cant_x_caja
            for _ in range(num_full_boxes):
                item_num += 1
                peso_paquete = peso_producto * cant_x_caja
                volumen_caja = op['VOLUMEN_m3']
                asignacion_cajas_list.append({
                    'ITEM': item_num,
                    'CAJA': desc_caja,
                    'CODIGO': codigo_producto,
                    'DESCRIPCION': descripcion_producto,
                    'CANTIDAD': cant_x_caja,
                    'PESO': peso_paquete,
                    'VOLUMEN': volumen_caja
                })
            remaining_quantity -= num_full_boxes * cant_x_caja

        # Sobrantes: tratar de usar una caja más pequeña
        if remaining_quantity > 0:
            opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'])
            caja_asignada = False
            for op in opciones:
                if op['CANTIDAD_X_CAJA'] >= remaining_quantity:
                    desc_caja = op['DESCRIPCION_CAJA']
                    item_num += 1
                    peso_paquete = peso_producto * remaining_quantity
                    volumen_caja = op['VOLUMEN_m3']
                    asignacion_cajas_list.append({
                        'ITEM': item_num,
                        'CAJA': desc_caja,
                        'CODIGO': codigo_producto,
                        'DESCRIPCION': descripcion_producto,
                        'CANTIDAD': remaining_quantity,
                        'PESO': peso_paquete,
                        'VOLUMEN': volumen_caja
                    })
                    remaining_quantity = 0
                    caja_asignada = True
                    break
            # Si todavía hay restos, usamos la caja más grande
            if not caja_asignada and remaining_quantity > 0:
                opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
                op = opciones[0]
                desc_caja = op['DESCRIPCION_CAJA']
                item_num += 1
                peso_paquete = peso_producto * remaining_quantity
                volumen_caja = op['VOLUMEN_m3']
                asignacion_cajas_list.append({
                    'ITEM': item_num,
                    'CAJA': desc_caja,
                    'CODIGO': codigo_producto,
                    'DESCRIPCION': descripcion_producto,
                    'CANTIDAD': remaining_quantity,
                    'PESO': peso_paquete,
                    'VOLUMEN': volumen_caja
                })
                remaining_quantity = 0
    else:
        print(f"No hay cajas definidas para el producto '{codigo_producto}'")

asignacion_cajas = pd.DataFrame(asignacion_cajas_list)

# ======================================
# 2) ORDENAR LAS CAJAS POR PESO
# ======================================
pesadas = asignacion_cajas[asignacion_cajas['PESO'] > 7.5].copy()
livianas = asignacion_cajas[asignacion_cajas['PESO'] <= 7.5].copy()

pesadas.sort_values(by='PESO', ascending=False, inplace=True)
livianas.sort_values(by='PESO', ascending=False, inplace=True)

asignacion_cajas = pd.concat([pesadas, livianas], ignore_index=True)

# Unimos para obtener dimensiones (ANCHO, ALTO, LARGO)
asignacion_cajas = asignacion_cajas.merge(
    dim_cajas[['DESCRIPCION', 'ANCHO_(mm)', 'ALTO_(mm)', 'LARGO_(mm)']],
    left_on='CAJA',
    right_on='DESCRIPCION',
    how='left',
    suffixes=('_prod', '_dim')
)

# Arreglamos columnas repetidas
if 'DESCRIPCION_prod' in asignacion_cajas.columns:
    asignacion_cajas.rename(columns={'DESCRIPCION_prod': 'DESCRIPCION'}, inplace=True)
if 'DESCRIPCION_dim' in asignacion_cajas.columns:
    asignacion_cajas.drop(columns=['DESCRIPCION_dim'], inplace=True, errors='ignore')

# ======================================
# 3) CREAR ITEMS PARA PY3DBP
#    width = LARGO_(mm)
#    height = ANCHO_(mm)
#    depth = ALTO_(mm)
# ======================================
all_items = []
for index, row in asignacion_cajas.iterrows():
    it = Item(
        name=f"ITEM_{int(row['ITEM'])}",
        width=float(row['LARGO_(mm)']),
        height=float(row['ANCHO_(mm)']),
        depth=float(row['ALTO_(mm)']),
        weight=float(row['PESO'])
    )
    # Asegurar rotaciones posibles
    it.rotation_type = RotationType.ALL
    all_items.append(it)

# ======================================
# 4) CREACIÓN DINÁMICA DE CAJONES
#    - Solo hay 2 tipos de cajones
#    - Se van agregando más según necesidad
# ======================================
# Guardamos en una lista (bin_types) la info de cada tipo de cajón
bin_types = []
for cod_cajon, datos_cajon in dimensiones_cajones.items():
    bin_types.append({
        'name':  datos_cajon['DESCRIPCION'],
        'width': float(datos_cajon['LARGO_(mm)']),
        'height': float(datos_cajon['ANCHO_(mm)']),
        'depth': float(datos_cajon['ALTO_(mm)']),
        'max_weight': 100000.0  # arbitrario
    })

# Función para crear un Bin (cajón) a partir de la info anterior
def crear_cajon(bin_info):
    return Bin(
        name=bin_info['name'],
        width=bin_info['width'],
        height=bin_info['height'],
        depth=bin_info['depth'],
        max_weight=bin_info['max_weight']
    )

def pack_con_bins_dinamicos(items, bin_types, max_iters=20):
    """
    Crea bins dinámicamente para empaquetar 'items'.
    Devuelve (packer, unfit_items) al final.
    """
    # Empezamos con 1 bin de cada tipo (2 bins si hay 2 tipos)
    bins_actuales = [crear_cajon(bt) for bt in bin_types]

    iter_count = 0
    while True:
        iter_count += 1

        # Creamos un packer nuevo en cada iteración
        packer = Packer()
        # Agregamos los bins actuales
        for b in bins_actuales:
            packer.add_bin(b)
        # Agregamos todos los items (py3dbp volverá a intentar empaquetarlos)
        for it in items:
            packer.add_item(it)

        # Empaquetar
        packer.pack(
            bigger_first=True,
            distribute_items=False,
            number_of_decimals=0
        )

        unfit = packer.unfit_items
        if not unfit:
            # Significa que todo cupo
            return packer, []
        else:
            # Si no caben todos y no excedimos iteraciones
            if iter_count >= max_iters:
                # Devolvemos lo que no cupo
                return packer, unfit

            # Agregamos un bin (cajón) más de *cada* tipo para el siguiente intento
            for bt in bin_types:
                bins_actuales.append(crear_cajon(bt))

# Empaquetamos dinámicamente
packer, unfit_items = pack_con_bins_dinamicos(all_items, bin_types, max_iters=20)

unfit_names = {it.name for it in unfit_items}

# ======================================
# 5) RECOPILAR INFORMACIÓN DE BIN / ITEM
# ======================================
item_to_cajon = {}
cajon_counter = 0

# packer.bins ya es la lista final de contenedores usados
# Sin embargo, ojo que py3dbp, por defecto, mantiene TODOS los bins,
# incluso los que quedaron vacíos sin items.
for b in packer.bins:
    if b.items:
        # Solo incrementamos el contador si el bin se usó realmente
        cajon_counter += 1
        descripcion_cajon = b.name
        for it in b.items:
            item_name = it.name
            item_to_cajon[item_name] = (
                descripcion_cajon, 
                cajon_counter,
                it.position, 
                it.width, 
                it.height, 
                it.depth
            )

def asignar_cajon(item_id):
    key = f"ITEM_{item_id}"
    if key in unfit_names:
        return ("SIN ASIGNAR", None)
    elif key in item_to_cajon:
        return (item_to_cajon[key][0], item_to_cajon[key][1])
    else:
        return ("SIN ASIGNAR", None)

asignacion_cajas['DESCRIPCION DE CAJON'] = asignacion_cajas['ITEM'].apply(lambda x: asignar_cajon(x)[0])
asignacion_cajas['# DE CAJON'] = asignacion_cajas['ITEM'].apply(lambda x: asignar_cajon(x)[1])

# Exportar resultados a Excel
asignacion_cajas.to_excel("asignacion_cajas.xlsx", index=False)
print("Archivo exportado: asignacion_cajas.xlsx")

if unfit_names:
    print("Algunos ítems NO se asignaron, incluso tras bins dinámicos.")
else:
    print("Todos los ítems fueron asignados con bins dinámicos.")

# ======================================
# 6) VISUALIZACIÓN CON PLOTLY
# ======================================

if not os.path.exists('cajones_3d'):
    os.makedirs('cajones_3d')

colors = [
    'blue', 'red', 'green', 'orange', 'purple', 'yellow', 
    'cyan', 'magenta', 'lime', 'pink'
]

# Índices para plotly (caras de cada cubo)
i_faces = [0,0,4,4,0,0,1,1,0,0,3,3]
j_faces = [1,2,5,6,3,7,2,6,1,5,2,6]
k_faces = [2,3,6,7,7,4,6,5,5,4,6,7]

# Vamos a enumerar sólo los bins que se usaron con items
plot_bin_counter = 0
for b in packer.bins:
    if not b.items:
        continue

    plot_bin_counter += 1
    fig = go.Figure()

    w = b.width   # X = width = LARGO
    h = b.height  # Y = height = ANCHO
    d = b.depth   # Z = depth = ALTO

    # Coordenadas del cajón
    X_cajon = [0,    w,    w,    0,    0,    w,    w,    0]
    Y_cajon = [0,    0,    h,    h,    0,    0,    h,    h]
    Z_cajon = [0,    0,    0,    0,    d,    d,    d,    d]

    fig.add_trace(go.Mesh3d(
        x=X_cajon,
        y=Y_cajon,
        z=Z_cajon,
        i=i_faces, j=j_faces, k=k_faces,
        color='black',
        opacity=0.95,
        flatshading=True,
        name=f'Cajón {plot_bin_counter}'
    ))

    color_index = 0

    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth

        X = [ix, ix+iw, ix+iw, ix, ix, ix+iw, ix+iw, ix]
        Y = [iy, iy, iy+ih, iy+ih, iy, iy, iy+ih, iy+ih]
        Z = [iz, iz, iz, iz, iz+id_, iz+id_, iz+id_, iz+id_]

        # Sacamos el ITEM_XX
        item_id = int(it.name.split('_')[1])
        row_item = asignacion_cajas[asignacion_cajas['ITEM'] == item_id].iloc[0]

        hover_text = (f"ITEM: {item_id}<br>"
                      f"CODIGO: {row_item['CODIGO']}<br>"
                      f"DESCRIPCION: {row_item['DESCRIPCION']}<br>"
                      f"CANTIDAD: {row_item['CANTIDAD']}<br>"
                      f"PESO: {row_item['PESO']:.2f} kg")

        fig.add_trace(go.Mesh3d(
            x=X, y=Y, z=Z,
            i=i_faces, j=j_faces, k=k_faces,
            color=colors[color_index % len(colors)],
            opacity=1,
            flatshading=True,
            name=f"Item {item_id}",
            hovertext=hover_text,
            hoverinfo='text'
        ))
        color_index += 1

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm) - LARGO',
            yaxis_title='Y (mm) - ANCHO',
            zaxis_title='Z (mm) - ALTO',
            aspectmode='data'
        ),
        title=f'Representación 3D del Cajón {plot_bin_counter}'
    )

    fig.write_html(f'cajones_3d/cajon_{plot_bin_counter}.html')

print("Visualizaciones 3D generadas en la carpeta cajones_3d.")

# Verificación final para detectar ítems que sobresalen
for b in packer.bins:
    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth
        # si se pasa en X, Y o Z
        if (ix + iw > b.width) or (iy + ih > b.height) or (iz + id_ > b.depth):
            print(f"El ítem {it.name} sobresale del cajón {b.name}")
