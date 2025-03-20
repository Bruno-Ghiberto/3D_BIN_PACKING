
def process_packing_list():
    import re
    import pandas as pd
    import numpy as np

    # ======================================
    # 1. EJEMPLO DE LECTURA DE EXCEL
    # ======================================
    archivo_packing_list = r'C:\Users\bghiberto\source\repos\3D_BIN_PACKING\DATASETS\PACKING LIST-11.xlsx'
    archivo_dimensiones = r'C:\Users\bghiberto\source\repos\3D_BIN_PACKING\DATASETS\DIMENSIONES CAJAS-NORMALIZADO.xlsx'
    archivo_pesos = r'C:\Users\bghiberto\source\repos\3D_BIN_PACKING\DATASETS\PESO_P.T.xlsx'

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

    # ======================================
    # 2. GENERACIÓN DE LA RELACIÓN PRODUCTO -> CAJAS POSIBLES
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
            return 1.0

    # ======================================
    # 3. CREAR EL DATAFRAME 'asignacion_cajas'
    # ======================================
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

            # 3a. Llenar cajas completas
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

            # 3b. Sobran unidades => buscar caja que las soporte
            if remaining_quantity > 0:
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
    # 4. AÑADIMOS DIMENSIONES DE LA CAJA
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
    # 5. MODIFICACIONES REQUERIDAS
    # ======================================
    asignacion_cajas.rename(columns={
        'ANCHO_(mm)': 'W',
        'ALTO_(mm)': 'H',
        'LARGO_(mm)': 'L'
    }, inplace=True)

    asignacion_cajas['CAJA'] = asignacion_cajas['CAJA'].str.replace(
        r'CAJA Nº (\d+)',
        r'CAJA #\1',
        regex=True
    )

    # ======================================
    # 6. EXPORTAR A CSV
    # ======================================
    asignacion_cajas.to_excel(r"C:\Users\bghiberto\source\repos\3D_BIN_PACKING\DATASETS\asignacion_cajas_final-_-.xlsx", index=False)
    print("Archivo exportado: asignacion_cajas_final.csv")
    print(asignacion_cajas.head())


def build_packing_dataframe(asignacion_cajas, decoder):
    import pandas as pd
    """
    Crea un nuevo DataFrame (df_final) a partir de asignacion_cajas,
    agregando:
      - '#_DE_CAJON': número de bin donde se colocó la caja
      - 'DESCRIPCION_DE_CAJON': algún nombre o etiqueta del bin
      - 'X0','Y0','Z0','X1','Y1','Z1': coordenadas min/max de la caja en 3D

    :param asignacion_cajas: DataFrame original con columnas 
                             [ITEM, CAJA, CODIGO, DESCRIPCION, CANTIDAD, ...].
    :param decoder: objeto PlacementProcedure ya ejecutado 
                    (con Bins, load_items, etc.)
    :return: df_final con las columnas originales + '#_DE_CAJON',
             'DESCRIPCION_DE_CAJON', y X0..Z1.
    """

    # Copiamos el asignacion_cajas para no alterar el original
    df_final = asignacion_cajas.copy()

    # Preparamos columnas vacías para cajón
    df_final["#_DE_CAJON"] = None
    df_final["DESCRIPCION_DE_CAJON"] = None

    # Preparamos columnas vacías para coordenadas
    df_final["X0"] = None
    df_final["Y0"] = None
    df_final["Z0"] = None
    df_final["X1"] = None
    df_final["Y1"] = None
    df_final["Z1"] = None

    # BPS es un array de índices 0..(N-1) que indica el orden de empaque
    BPS = decoder.BPS  
    index_in_sequence = 0

    # Recorremos cada BIN (cajón) usado
    for bin_index, bin_obj in enumerate(decoder.Bins, start=1):
        desc_cajon = f"CAJON_{bin_index}"

        if not bin_obj.load_items:
            continue  # si el bin está vacío, no hacemos nada

        # Por cada caja en bin_obj.load_items (en el orden que se fueron agregando)
        for coords in bin_obj.load_items:
            # coords => [[x0,y0,z0],[x1,y1,z1]]
            original_index = BPS[index_in_sequence]
            index_in_sequence += 1

            # Recuperamos min / max
            (x0, y0, z0) = coords[0]
            (x1, y1, z1) = coords[1]

            # Asignamos datos en df_final
            df_final.at[original_index, "#_DE_CAJON"] = bin_index
            df_final.at[original_index, "DESCRIPCION_DE_CAJON"] = desc_cajon

            # Guardamos las coordenadas
            df_final.at[original_index, "X0"] = x0
            df_final.at[original_index, "Y0"] = y0
            df_final.at[original_index, "Z0"] = z0
            df_final.at[original_index, "X1"] = x1
            df_final.at[original_index, "Y1"] = y1
            df_final.at[original_index, "Z1"] = z1

    return df_final

