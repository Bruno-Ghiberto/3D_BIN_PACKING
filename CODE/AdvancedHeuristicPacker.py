import pandas as pd
import itertools

# --------------------------------
# 1) Basic Data Structures
# --------------------------------

class Box:
    """
    A box with original (w,h,l) but we can rotate it in up to 6 ways.
    """
    def __init__(self, item_id, w, h, l):
        self.item_id = item_id
        self.w = w
        self.h = h
        self.l = l
    
    def all_orientations(self):
        """
        Return all 6 permutations (W,H,L).
        E.g. (w, h, l), (w, l, h), (h, w, l), etc.
        """
        dims = [self.w, self.h, self.l]
        # set(...) to remove duplicates if some dims are equal
        perms = set(itertools.permutations(dims, 3))
        return list(perms)

    def volume(self):
        return self.w * self.h * self.l

class FreeRectangle:
    """
    A rectangle in the 2D plane of a shelf,
    defined by (x, y, width, depth).
    'z_offset' is inherited from the shelf.
    """
    def __init__(self, x, y, width, depth):
        self.x = x
        self.y = y
        self.width = width   # along X-axis
        self.depth = depth   # along Y-axis

    def __repr__(self):
        return f"<Rect x={self.x} y={self.y} w={self.width} d={self.depth}>"


class Shelf2D:
    """
    A "horizontal layer" at some z_offset in the bin.
    Maintains a list of free 2D rectangles. Height is
    determined by the tallest box placed in this layer.
    """
    def __init__(self, z_offset, bin_length, bin_width):
        self.z_offset = z_offset
        self.bin_length = bin_length  # dimension along X
        self.bin_width  = bin_width   # dimension along Y

        # Initially, the entire shelf is one big free rectangle
        initial_rect = FreeRectangle(x=0, y=0,
                                     width=bin_length,
                                     depth=bin_width)
        self.free_rects = [initial_rect]

        self.current_height = 0  # track the max height used in this shelf

        # For visualization/records
        self.placements = []  # list of dict with placement info

    def try_place_box(self, box):
        """
        Attempt to place 'box' in any free rectangle in any orientation.
        If successful, update leftover rectangles and shelf height, return True.
        If not, return False.
        """
        possible_orientations = box.all_orientations()

        for rect_idx, rect in enumerate(self.free_rects):
            for (w, h, d) in possible_orientations:
                # We try to place dimension (w,d) in rect.width x rect.depth
                # The height is 'h' -> if we place it, shelf's 'height' might increase
                if w <= rect.width and d <= rect.depth:
                    # We can place the box here
                    x0 = rect.x
                    y0 = rect.y
                    x1 = x0 + w
                    y1 = y0 + d
                    z0 = self.z_offset
                    z1 = z0 + h

                    # Record the placement
                    self.placements.append({
                        "BOX_ID":   box.item_id,
                        "x0":       x0,
                        "y0":       y0,
                        "z0":       z0,
                        "x1":       x1,
                        "y1":       y1,
                        "z1":       z1
                    })

                    # Update shelf height if needed
                    if h > self.current_height:
                        self.current_height = h

                    # Split the used rectangle
                    self._split_free_rectangle(rect_idx, w, d)
                    return True
        # If we can't place it in any free rect with any orientation, fail
        return False

    def _split_free_rectangle(self, rect_idx, used_w, used_d):
        """
        Perform a guillotine cut on the rectangle at self.free_rects[rect_idx]
        after placing an item of (used_w, used_d) in its lower-left corner.
        """
        rect = self.free_rects[rect_idx]
        # The placed box is at (rect.x, rect.y) with width=used_w, depth=used_d

        # We'll remove the used rectangle
        del self.free_rects[rect_idx]

        # We produce up to two leftover rectangles (right split and top split):
        # 1) The space to the RIGHT of the placed box
        remaining_width = rect.width - used_w
        if remaining_width > 0:
            r1 = FreeRectangle(
                x=rect.x + used_w,
                y=rect.y,
                width=remaining_width,
                depth=rect.depth
            )
            self.free_rects.append(r1)

        # 2) The space ABOVE the placed box
        remaining_depth = rect.depth - used_d
        if remaining_depth > 0:
            r2 = FreeRectangle(
                x=rect.x,
                y=rect.y + used_d,
                width=used_w,     # only the used width is cut out
                depth=remaining_depth
            )
            self.free_rects.append(r2)

        # Optionally, you can apply further merges or advanced logic
        # to keep the free_rects list minimal or combined.

class SingleBin2DPacker:
    """
    Manages 2D-based shelf packing within one bin (for the vertical dimension).
    """
    def __init__(self, bin_length, bin_width, bin_height):
        self.bin_length = bin_length
        self.bin_width  = bin_width
        self.bin_height = bin_height

        self.shelves = []
        self.used_height = 0  # sum of shelf heights so far

    def try_place_box(self, box):
        """
        Try to place 'box' in existing shelves, or open a new shelf if needed.
        Return True if successfully placed, False otherwise (bin is "full").
        """
        # First, check the existing shelves
        for shelf in self.shelves:
            if shelf.try_place_box(box):
                return True

        # If not placed, try adding a new shelf
        # But check if there's enough vertical space for the new shelf
        # 'current shelf top' is self.used_height
        # We'll place a new shelf at z_offset = self.used_height
        # We don't yet know how tall it might become, but let's see:
        # If the box height is h, that means final top might be used_height + h
        # We'll just attempt placing in a fresh shelf. If it fits horizontally,
        # that shelf's final height is the box's height, but we have to ensure
        # used_height + h <= bin_height.
        for (w, h, d) in box.all_orientations():
            if (self.used_height + h) <= self.bin_height:
                # We can create a fresh shelf
                new_shelf = Shelf2D(
                    z_offset=self.used_height,
                    bin_length=self.bin_length,
                    bin_width=self.bin_width
                )
                self.shelves.append(new_shelf)
                # Attempt placement in the new shelf with that orientation
                success = new_shelf.try_place_box(box)
                if success:
                    # If the shelf's current_height is bigger than what we
                    # had planned, update used_height.
                    # But we need to figure out how tall that shelf ended up.
                    if new_shelf.current_height > 0:
                        self.used_height += new_shelf.current_height
                    return True
                else:
                    # If it fails with that orientation, we remove the shelf
                    self.shelves.pop()
                    return False
        # If no orientation can fit even in a fresh shelf, we give up
        return False

    def get_placements(self):
        """
        Return a list of all placed items in this bin, with their coordinates.
        """
        placements = []
        for shelf_id, shelf in enumerate(self.shelves, start=1):
            for record in shelf.placements:
                record_copy = dict(record)  # copy to avoid mutation
                record_copy["SHELF_ID"] = shelf_id
                placements.append(record_copy)
        return placements

class MultiBin2DPacker:
    """
    Multiple-bin manager. 
    First-Fit approach across bins: for each box, try bins in order;
    if it doesn't fit, create a new bin (if possible).
    """
    def __init__(self, bin_length, bin_width, bin_height, items):
        self.bin_length = bin_length
        self.bin_width  = bin_width
        self.bin_height = bin_height
        self.items = items

        self.bins = []

    def pack_items(self):
        # Sort descending by volume
        self.items.sort(key=lambda b: b.volume(), reverse=True)

        for box in self.items:
            placed = False
            # Try existing bins first
            for b in self.bins:
                if b.try_place_box(box):
                    placed = True
                    break
            if not placed:
                # Open a new bin
                new_bin = SingleBin2DPacker(
                    self.bin_length,
                    self.bin_width,
                    self.bin_height
                )
                success = new_bin.try_place_box(box)
                if success:
                    self.bins.append(new_bin)
                else:
                    # Box doesn't fit even in an empty bin => skip or report error
                    print(f"Box {box.item_id} cannot fit in a brand new bin. Skipped.")

    def get_placements_dataframe(self):
        """
        Returns DataFrame with columns:
          [BIN_ID, SHELF_ID, BOX_ID, x0,y0,z0, x1,y1,z1]
        """
        records = []
        for bin_idx, single_bin in enumerate(self.bins, start=1):
            bin_placements = single_bin.get_placements()
            for p in bin_placements:
                p["BIN_ID"] = bin_idx
                records.append(p)
        return pd.DataFrame(records)

