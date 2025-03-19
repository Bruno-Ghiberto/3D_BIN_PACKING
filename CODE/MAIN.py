# main.py

import os
import pandas as pd
import Utils
from Plotter import HeuristicPlotter
from AdvancedHeuristicPacker import Box, MultiBin2DPacker

def load_items_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    items = []
    for _, row in df.iterrows():
        w = float(row['W'])
        h = float(row['H'])
        l = float(row['L'])
        item_id = row['ITEM']
        quantity = int(row.get('CANTIDAD', 1))

        for _ in range(quantity):
            items.append(Box(item_id=item_id, w=w, h=h, l=l))

    return items

def main():
    Utils.process_packing_list()  # Generate asignacion_cajas_final.csv if needed

    CSV_PATH = r"C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D-BPP\DATASETS\asignacion_cajas_final.csv"
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    # 1) Load items for packing
    items = load_items_from_csv(CSV_PATH)
    print(f"Loaded {len(items)} total boxes from CSV.")

    BIN_LENGTH = 860
    BIN_WIDTH  = 890
    BIN_HEIGHT = 1040

    # 2) Pack the items
    packer = MultiBin2DPacker(bin_length=BIN_LENGTH,
                              bin_width=BIN_WIDTH,
                              bin_height=BIN_HEIGHT,
                              items=items)
    packer.pack_items()

    # 3) Basic packing results => [BIN_ID, SHELF_ID, BOX_ID, x0, y0, z0, x1, y1, z1]
    df_placements = packer.get_placements_dataframe()
    total_placed = len(df_placements)
    print(f"Placed {total_placed} boxes out of {len(items)} total.")

    # ### NEW ### Load the original CSV again, so we can merge CAJA/DESCRIPCION
    df_original = pd.read_csv(CSV_PATH)  
    # df_original has columns like: [ITEM, CAJA, DESCRIPCION, CANTIDAD, W, H, L, ...]

    # ### NEW ### Merge them on BOX_ID == ITEM
    df_merged = df_placements.merge(
        df_original[['ITEM','CAJA','DESCRIPCION','CANTIDAD']],
        how='left',
        left_on='BOX_ID',
        right_on='ITEM'
    )
    # Now df_merged has: 
    # [BIN_ID, SHELF_ID, BOX_ID, x0..z1, ITEM, CAJA, DESCRIPCION, CANTIDAD]

    # Optionally drop the duplicate "ITEM" column
    df_merged.drop(columns=['ITEM'], inplace=True)

    # 4) Save the final merged data (with CAJA, DESCRIPCION, etc.)
    output_csv = r"C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D-BPP\DATASETS\placements_result.csv"
    df_merged.to_csv(output_csv, index=False)
    print(f"Saved merged placements to: {output_csv}")

    # 5) Plot
    df_for_plot = pd.read_csv(output_csv)  # re-load the merged file

    plotter = HeuristicPlotter()
    output_dir = r"C:\Users\bghiberto\Documents\Bruno Ghiberto\IA EN LOGÍSTICA\3D-BPP\DATASETS"
    plotter.plot_all_bins(
        df_placements=df_for_plot,
        bin_length=BIN_LENGTH,
        bin_width=BIN_WIDTH,
        bin_height=BIN_HEIGHT,
        output_folder=output_dir
    )

if __name__ == "__main__":
    main()
