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
        # Peso por defecto si no se encuentra
        return 1.0

# Crear el dataframe asignacion_cajas
asignacion_cajas_list = []
item_num = 0

for _, producto_row in packing_list.iterrows():
    codigo_producto = str(producto_row['CODIGO']).strip()
    descripcion_producto = producto_row['DESCRIPCION']
    cantidad_producto = int(producto_row['CANTIDAD'])

    if codigo_producto in producto_a_cajas:
        opciones = producto_a_cajas[codigo_producto]
        for op in opciones:
            desc_caja = op['DESCRIPCION_CAJA']
            if desc_caja in dimensiones_cajas:
                op['VOLUMEN_m3'] = dimensiones_cajas[desc_caja]['VOLUMEN_m3']
            else:
                op['VOLUMEN_m3'] = float('inf')

        opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
        remaining_quantity = cantidad_producto
        peso_producto = obtener_peso_producto(codigo_producto)

        # Empaquetar cajas completas
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

        # Sobrantes
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

            if not caja_asignada and remaining_quantity > 0:
                # Volver a la mayor
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

# Ordenar por peso
pesadas = asignacion_cajas[asignacion_cajas['PESO'] > 7.5].copy()
livianas = asignacion_cajas[asignacion_cajas['PESO'] <= 7.5].copy()
pesadas.sort_values(by='PESO', ascending=False, inplace=True)
livianas.sort_values(by='PESO', ascending=False, inplace=True)
asignacion_cajas = pd.concat([pesadas, livianas], ignore_index=True)

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

# Crear ítems para py3dbp
# Importante: mantener la consistencia de ejes
# width = LARGO_(mm)
# height = ANCHO_(mm)
# depth = ALTO_(mm)
all_items = []
for index, row in asignacion_cajas.iterrows():
    it = Item(
        name=f"ITEM_{int(row['ITEM'])}",
        width=float(row['LARGO_(mm)']),
        height=float(row['ANCHO_(mm)']),
        depth=float(row['ALTO_(mm)']),
        weight=float(row['PESO'])
    )
    # Restringir rotaciones
    it.rotation_type = RotationType.ALL
    all_items.append(it)

packer = Packer()

num_cajones_por_tipo = 100
for codigo_cajon, datos_cajon in dimensiones_cajones.items():
    descripcion_cajon = datos_cajon['DESCRIPCION']
    largo_cajon = datos_cajon['LARGO_(mm)']
    ancho_cajon = datos_cajon['ANCHO_(mm)']
    alto_cajon = datos_cajon['ALTO_(mm)']
    peso_maximo = 100000

    # Mantener el mismo orden de dimensiones que en ítems
    # width = largo_cajon
    # height = ancho_cajon
    # depth = alto_cajon
    for i in range(num_cajones_por_tipo):
        cajon = Bin(
            name=f"{descripcion_cajon}",
            width=float(largo_cajon),
            height=float(ancho_cajon),
            depth=float(alto_cajon),
            max_weight=peso_maximo
        )
        packer.add_bin(cajon)

for it in all_items:
    packer.add_item(it)

packer.pack(bigger_first=True, distribute_items=True, number_of_decimals=0)

unfit_names = {it.name for it in packer.unfit_items}

item_to_cajon = {}
cajon_counter = 0
for b in packer.bins:
    if b.items:
        cajon_counter += 1
        descripcion_cajon = b.name
        for it in b.items:
            item_name = it.name
            item_to_cajon[item_name] = (descripcion_cajon, cajon_counter, it.position, it.width, it.height, it.depth)

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

asignacion_cajas.to_excel("asignacion_cajas.xlsx", index=False)
print("Archivo exportado: asignacion_cajas.xlsx")

if unfit_names:
    print("Algunos ítems no se asignaron.")
else:
    print("Todos los ítems fueron asignados.")

# ======================================
# VISUALIZACIÓN CON PLOTLY
# ======================================

if not os.path.exists('cajones_3d'):
    os.makedirs('cajones_3d')

colors = [
    'blue', 'red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'lime', 'pink'
]

# Caras del cubo (ítem y cajón)
i_faces = [0,0,4,4,0,0,1,1,0,0,3,3]
j_faces = [1,2,5,6,3,7,2,6,1,5,2,6]
k_faces = [2,3,6,7,7,4,6,5,5,4,6,7]

for idx_b, b in enumerate(packer.bins, start=1):
    if not b.items:
        continue

    fig = go.Figure()

    w = b.width   # X = width = largo
    h = b.height  # Y = height = ancho
    d = b.depth   # Z = depth = alto

    X_cajon = [0,    w,    w,    0,    0,    w,    w,    0   ]
    Y_cajon = [0,    0,    h,    h,    0,    0,    h,    h   ]
    Z_cajon = [0,    0,    0,    0,    d,    d,    d,    d   ]

    fig.add_trace(go.Mesh3d(
        x=X_cajon,
        y=Y_cajon,
        z=Z_cajon,
        i=i_faces, j=j_faces, k=k_faces,
        color='black',
        opacity=0.95,
        flatshading=True,
        name=f'Cajón {idx_b}'
    ))

    color_index = 0

    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth

        X = [ix, ix+iw, ix+iw, ix, ix, ix+iw, ix+iw, ix]
        Y = [iy, iy, iy+ih, iy+ih, iy, iy, iy+ih, iy+ih]
        Z = [iz, iz, iz, iz, iz+id_, iz+id_, iz+id_, iz+id_]

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
        title=f'Representación 3D del Cajón {idx_b}'
    )

    fig.write_html(f'cajones_3d/cajon_{idx_b}.html')

print("Visualizaciones 3D generadas en la carpeta cajones_3d.")

# Verificación final de sobresalir
for b in packer.bins:
    for it in b.items:
        ix, iy, iz = it.position
        iw, ih, id_ = it.width, it.height, it.depth
        if ix + iw > b.width or iy + ih > b.height or iz + id_ > b.depth:
            print(f"El ítem {it.name} sobresale del cajón {b.name}")


