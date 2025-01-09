import os
import re
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model

# ======================================
# 1. FUNCIÓN PARA DETECTAR ENGRANAJES (opcional)
# ======================================
def es_engranaje(codigo_producto):
    patron = r'^1\.\d{3}1\.\d{7}$'
    return bool(re.match(patron, codigo_producto.strip()))

def obtener_peso_producto(codigo_producto, pesos_productos_dict):
    peso = pesos_productos_dict.get(codigo_producto)
    if peso is not None and not pd.isnull(peso):
        return float(peso)
    else:
        return 1.0  # valor por defecto

def build_dataframes(
    archivo_packing_list, 
    archivo_dimensiones, 
    archivo_pesos
):
    """
    Función que lee los excels y crea los dataframes 
    (packing_list, caja_producto, dim_cajas, tam_cajones, pesos_productos).
    """
    dim_cajas = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJAS')
    caja_producto = pd.read_excel(archivo_dimensiones, sheet_name='CAJA-PRODUCTO')
    tam_cajones = pd.read_excel(archivo_dimensiones, sheet_name='TAMAÑO-CAJONES')
    packing_list = pd.read_excel(archivo_packing_list, sheet_name='P.L.')
    pesos_productos = pd.read_excel(archivo_pesos, sheet_name='PRODUCTO-PESO')

    # Limpieza de columnas y normalización
    for df in [dim_cajas, caja_producto, tam_cajones, packing_list, pesos_productos]:
        df.columns = df.columns.str.strip()

    # Ajustes específicos
    dim_cajas['DESCRIPCION'] = dim_cajas['DESCRIPCION'].astype(str).str.strip()

    caja_cols = ['CAJA_OP_1', 'CAJA_OP_2', 'CAJA_OP_3']
    cantidad_cols = ['CANTIDAD x CAJA_OP_1', 'CANTIDAD_x_CAJA_OP_2', 'CANTIDAD_x_CAJA_OP_3']
    for col in caja_cols:
        if col in caja_producto.columns:
            caja_producto[col] = caja_producto[col].astype(str).str.strip()

    pesos_productos['CODIGO'] = pesos_productos['CODIGO'].astype(str).str.strip()
    pesos_productos['PESO(kg)'] = pd.to_numeric(pesos_productos['PESO(kg)'], errors='coerce')
    packing_list['CODIGO'] = packing_list['CODIGO'].astype(str).str.strip()

    return dim_cajas, caja_producto, tam_cajones, packing_list, pesos_productos


def build_asignacion_cajas(
    packing_list, 
    caja_producto, 
    dim_cajas, 
    pesos_productos
):
    """
    Lógica para generar el dataframe 'asignacion_cajas',
    donde cada fila representa una 'caja' a empacar (un ítem en py3dbp).
    """
    # Diccionarios para acceso rápido
    pesos_productos_dict = pesos_productos.set_index('CODIGO')['PESO(kg)'].to_dict()
    dimensiones_cajas = dim_cajas.set_index('DESCRIPCION').to_dict('index')

    # Diccionario: producto -> [opciones de cajas]
    producto_a_cajas = {}
    caja_cols = ['CAJA_OP_1', 'CAJA_OP_2', 'CAJA_OP_3']
    cantidad_cols = ['CANTIDAD x CAJA_OP_1', 'CANTIDAD_x_CAJA_OP_2', 'CANTIDAD_x_CAJA_OP_3']
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

    asignacion_cajas_list = []
    item_num = 0

    for _, producto_row in packing_list.iterrows():
        codigo_producto = str(producto_row['CODIGO']).strip()
        descripcion_producto = producto_row['DESCRIPCION']
        cantidad_producto = int(producto_row['CANTIDAD'])

        if codigo_producto in producto_a_cajas:
            opciones = producto_a_cajas[codigo_producto]
            # Insertar volumen, etc., si hace falta
            for op in opciones:
                desc_caja = op['DESCRIPCION_CAJA']
                if desc_caja in dimensiones_cajas:
                    op['VOLUMEN_m3'] = dimensiones_cajas[desc_caja]['VOLUMEN_m3']
                else:
                    op['VOLUMEN_m3'] = float('inf')
            
            # Ordenar por CANTIDAD_X_CAJA descendente
            opciones.sort(key=lambda x: x['CANTIDAD_X_CAJA'], reverse=True)
            remaining_quantity = cantidad_producto
            peso_producto = obtener_peso_producto(codigo_producto, pesos_productos_dict)

            # Llenar cajas completas
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
                        'PESO': peso_paquete
                    })
                remaining_quantity -= num_full_boxes * cant_x_caja

            # Sobran unidades
            if remaining_quantity > 0:
                # Ordenar por CANTIDAD_X_CAJA ascendente
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
                            'PESO': peso_paquete
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
                        'PESO': peso_paquete
                    })
                    remaining_quantity = 0
        else:
            print(f"No hay cajas definidas para el producto '{codigo_producto}'")

    asignacion_cajas = pd.DataFrame(asignacion_cajas_list)
    # Ahora unimos con dimensiones de la CAJA
    asignacion_cajas = asignacion_cajas.merge(
        dim_cajas[['DESCRIPCION', 'ANCHO_(mm)', 'ALTO_(mm)', 'LARGO_(mm)']],
        left_on='CAJA',
        right_on='DESCRIPCION',
        how='left'
    )
    if 'DESCRIPCION' in asignacion_cajas.columns.duplicated():
        # Manejo de colisiones de columnas (si se da)
        pass

    # Reordenar / renombrar si hace falta
    return asignacion_cajas


# ======================================
# 2. ESQUELETO DEL SOLVER OR-TOOLS
# ======================================

def solve_3d_bin_packing(asignacion_cajas, bin_dims, max_bins):
    """
    Dado el dataframe asignacion_cajas con la info de cada 'ítem' (caja a empacar),
    y las dimensiones del bin (W,H,D), resuelve con OR-Tools CP-SAT 
    el problema de bin packing 3D.
    
    bin_dims: (W, H, D) en mm
    max_bins: cantidad máxima de bins (por ejemplo 9, si sabemos que no usaremos más)
    """
    model = cp_model.CpModel()

    # Preparamos la data
    # items = [ (idx, w_i, h_i, d_i, peso_i, etc...) ]
    items = []
    for _, row in asignacion_cajas.iterrows():
        item_id = int(row['ITEM'])
        ancho = float(row['ANCHO_(mm)'])
        alto  = float(row['ALTO_(mm)'])
        largo = float(row['LARGO_(mm)'])
        peso  = float(row['PESO'])  # si quieres usarlo
        # Podrías reordenar ancho, alto, largo a conveniencia.
        # De momento, supongamos (x,y,z) = (ancho, alto, largo).
        # Lo importante es ser consistente luego.
        items.append( (item_id, ancho, alto, largo, peso) )

    # Vars de asignación: assign[i][b] => bool
    n = len(items)
    assign = []
    for i in range(n):
        row = []
        for b in range(max_bins):
            row.append(model.NewBoolVar(f'assign_{i}_{b}'))
        assign.append(row)

    # Vars de "bin usado": used[b] => bool
    used = [model.NewBoolVar(f'used_{b}') for b in range(max_bins)]

    # Vars de posición (x_i_b, y_i_b, z_i_b) => int
    # Limitadas por el tamaño del bin
    W, H, D = bin_dims
    x = []
    y = []
    z = []
    for i in range(n):
        xrow, yrow, zrow = [], [], []
        for b in range(max_bins):
            # En mm
            xv = model.NewIntVar(0, W, f'x_{i}_{b}')
            yv = model.NewIntVar(0, H, f'y_{i}_{b}')
            zv = model.NewIntVar(0, D, f'z_{i}_{b}')
            xrow.append(xv)
            yrow.append(yv)
            zrow.append(zv)
        x.append(xrow)
        y.append(yrow)
        z.append(zrow)

    # 1) Cada ítem en un bin
    for i in range(n):
        model.Add(sum(assign[i][b] for b in range(max_bins)) == 1)

    # 2) Si un item se asigna a un bin => ese bin está usado
    for i in range(n):
        for b in range(max_bins):
            model.Add(assign[i][b] <= used[b])

    # 3) No sobresalir
    for i in range(n):
        _, w_i, h_i, d_i, _ = items[i]
        for b in range(max_bins):
            # x_i + w_i <= W, etc.
            model.Add(x[i][b] + int(w_i) <= W).OnlyEnforceIf(assign[i][b])
            model.Add(y[i][b] + int(h_i) <= H).OnlyEnforceIf(assign[i][b])
            model.Add(z[i][b] + int(d_i) <= D).OnlyEnforceIf(assign[i][b])

    # 4) No solapamiento (disyunción) => generará bastantes constraints
    for i in range(n):
        id_i, w_i, h_i, d_i, _ = items[i]
        for j in range(i+1, n):
            id_j, w_j, h_j, d_j, _ = items[j]
            for b in range(max_bins):
                # both_in_b => AND(assign[i][b], assign[j][b])
                both_in_b = model.NewBoolVar(f'both_{i}_{j}_{b}')
                model.AddBoolAnd([assign[i][b], assign[j][b]]).OnlyEnforceIf(both_in_b)
                model.AddBoolOr([assign[i][b].Not(), assign[j][b].Not()]).OnlyEnforceIf(both_in_b.Not())

                # now we create the disjunction => no_overlap_b
                no_overlap_b = model.NewBoolVar(f'no_overlap_{i}_{j}_{b}')

                # 6 condiciones: (x_i + w_i <= x_j) OR (x_j + w_j <= x_i)
                #               (y_i + h_i <= y_j) OR (y_j + h_j <= y_i)
                #               (z_i + d_i <= z_j) OR (z_j + d_j <= z_i)
                conds = []

                cond1 = model.NewBoolVar('')
                model.Add( x[i][b] + int(w_i) <= x[j][b] ).OnlyEnforceIf(cond1)
                conds.append(cond1)

                cond2 = model.NewBoolVar('')
                model.Add( x[j][b] + int(w_j) <= x[i][b] ).OnlyEnforceIf(cond2)
                conds.append(cond2)

                cond3 = model.NewBoolVar('')
                model.Add( y[i][b] + int(h_i) <= y[j][b] ).OnlyEnforceIf(cond3)
                conds.append(cond3)

                cond4 = model.NewBoolVar('')
                model.Add( y[j][b] + int(h_j) <= y[i][b] ).OnlyEnforceIf(cond4)
                conds.append(cond4)

                cond5 = model.NewBoolVar('')
                model.Add( z[i][b] + int(d_i) <= z[j][b] ).OnlyEnforceIf(cond5)
                conds.append(cond5)

                cond6 = model.NewBoolVar('')
                model.Add( z[j][b] + int(d_j) <= z[i][b] ).OnlyEnforceIf(cond6)
                conds.append(cond6)

                # no_overlap_b => OR(conds)
                model.AddBoolOr(conds).OnlyEnforceIf(no_overlap_b)
                # Si both_in_b, => no_overlap_b
                model.AddImplication(both_in_b, no_overlap_b)

    # 5) Objetivo: Minimizar bins usados
    model.Minimize(sum(used[b] for b in range(max_bins)))

    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 300  # 5 min, ajusta a gusto
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Status:', solver.StatusName(status))
        used_bins = 0
        item_to_bin = {}
        coords_solution = {}  # Para guardar (x,y,z)
        for b in range(max_bins):
            if solver.Value(used[b]) == 1:
                used_bins += 1
                # Recorremos items
                for i in range(n):
                    if solver.Value(assign[i][b]) == 1:
                        item_id = items[i][0]
                        item_to_bin[item_id] = b
                        sol_x = solver.Value(x[i][b])
                        sol_y = solver.Value(y[i][b])
                        sol_z = solver.Value(z[i][b])
                        coords_solution[item_id] = (sol_x, sol_y, sol_z)
        print(f'Bins usados: {used_bins} de {max_bins}')
        return item_to_bin, coords_solution, used_bins
    else:
        print("No se encontró solución factible.")
        return None, None, None


# ======================================
# 3. PUNTO DE ENTRADA (MAIN)
# ======================================

if __name__ == "__main__":
    # Rutas de archivos (ajusta a tu entorno):
    archivo_packing_list = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D_BIN_PACKING__OR-Tools\PACKING LIST.xlsx'
    archivo_dimensiones  = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D_BIN_PACKING__OR-Tools\DIMENSIONES CAJAS-NORMALIZADO.xlsx'
    archivo_pesos        = r'C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D_BIN_PACKING__OR-Tools\PESO_P.T.xlsx'

    # 1) Leemos dataframes
    dim_cajas, caja_producto, tam_cajones, packing_list, pesos_productos = build_dataframes(
        archivo_packing_list, 
        archivo_dimensiones, 
        archivo_pesos
    )

    # 2) Creamos el DF asignacion_cajas (cada "paquete" a empacar)
    asignacion_cajas = build_asignacion_cajas(
        packing_list, 
        caja_producto, 
        dim_cajas, 
        pesos_productos
    )

    # 3) Supongamos que tenemos un único tipo de cajón => sacamos sus dimensiones (ej. "CODIGO"= X?)
    #    Si en tu Excel "tam_cajones" tienes un cajón con CODIGO "CAJON_NORMAL", obtén su W,H,D
    #    O si ya sabes que tus bins son 9 cajones con dim: 890 x 860 x 1040 mm:
    bin_dims = (890, 860, 1040)

    # En tu caso real, se usaron 9 cajones
    max_bins = 9

    # 4) Llamamos al solver
    item_to_bin, coords_solution, used_bins = solve_3d_bin_packing(
        asignacion_cajas,
        bin_dims,
        max_bins
    )

    # 5) Si hay solución, podemos fusionarla de vuelta con asignacion_cajas
    if item_to_bin is not None:
        asignacion_cajas['BIN'] = asignacion_cajas['ITEM'].map(item_to_bin)
        # Guardar posiciones (x,y,z) si lo necesitas
        asignacion_cajas['POSICION'] = asignacion_cajas['ITEM'].map(coords_solution)
        
        asignacion_cajas.to_excel("asignacion_cajas_con_bins.xlsx", index=False)
        print("Exportado asignacion_cajas_con_bins.xlsx")

