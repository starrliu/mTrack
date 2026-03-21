import csv
import os
from tqdm import tqdm
import pandas as pd

class Tracklet:
    def __init__(self, objid) -> None:
        self.object_id = objid

        self.frames = []
        self.x_lst = []
        self.y_lst = []
        self.w_lst = []
        self.h_lst = []

        self.matched_points = list()    # (frame, tagid)
        self.gc_points = list()        # (frame)
        self.bc_point = None

        self._segments = list()
        self.assignments = list()

    def add_frame(self, frame, x, y, w, h):
        self.frames.append(frame)
        self.x_lst.append(x)
        self.y_lst.append(y)
        self.w_lst.append(w)
        self.h_lst.append(h)

    def add_matched_point(self, frame, tagid):
        self.matched_points.append((frame, tagid))
    
    def add_gc_point(self, frame):
        self.gc_points.append(frame)

    def add_bc_point(self, tagid):
        self.bc_point = tagid

    def generate_assignments(self):
        # Generate the assignments based on matched_points and gc_points
        # 1. Segment the frames into different slots by gc_points
        # 2. Assign the tagid to the slot by the matched_points
        # 3. Assign the last slot by the bc_point

        # Step 1: Segment the frames
        segments = []
        if len(self.frames) == 0:
            return
        
        cur_frame = self.frames[0]
        idx_gc = 0
        while cur_frame <= self.frames[-1]:
            if idx_gc < len(self.gc_points):
                cur_gc = self.gc_points[idx_gc]
                if cur_frame <= cur_gc:
                    segments.append((cur_frame, min(cur_gc, self.frames[-1])))
                    cur_frame = cur_gc + 1
                    idx_gc += 1
                else:
                    print("idx_gc > cur_frame")
                    idx_gc += 1
            else:
                segments.append((cur_frame, self.frames[-1]))
                break
        self._segments = segments

        # Step 2: Assign the tagid to the slot
        last_tagid = None
        for idx in range(len(segments)):
            start_f, end_f = segments[idx]

            is_matched = False
            for frame, cur_tagid in self.matched_points:
                if frame >= start_f and frame <= end_f:
                    tagid = cur_tagid
                    last_tagid = tagid
                    is_matched = True
                    self.assignments.append((start_f, end_f, tagid))
                    break
            
            if not is_matched:
                if last_tagid is not None:
                    self.assignments.append((start_f, end_f, last_tagid))

        # Step 3: Assign the last slot by the bc_point
        if self.bc_point is not None:
            self.assignments.append((segments[-1][0], segments[-1][1], self.bc_point))
       
    def add_assignment(self, start_f, end_f, tagid):
        if start_f < self.start_frame:
            start_f = self.start_frame
        if end_f > self.end_frame:
            end_f = self.end_frame

        self.assignments.append((start_f, end_f, tagid))

    def get_identity(self, frame):
        for start_f, end_f, tagid in self.assignments:
            if frame >= start_f and frame <= end_f:
                return tagid
        return None

    @property
    def start_frame(self):
        if len(self.frames) == 0:
            return None
        return self.frames[0]
    
    @property
    def end_frame(self):
        if len(self.frames) == 0:
            return None
        return self.frames[-1]
    
    def __str__(self) -> str:
        output = f"Tracklet {self.object_id}\n"
        output += f"Lifetime: {self.start_frame} - {self.end_frame}\n"
        output += f"Matched points: {self.matched_points}\n"
        output += f"GC points: {self.gc_points}\n"
        output += f"BC point: {self.bc_point}\n"
        output += f"Segments: {self._segments}\n"
        output += f"Assignments: {self.assignments}\n"
        return output

class GtFileGenerator:
    def __init__(self, tags2id: dict[bytes, int], dumpdir, gtfile):
        self.tags2id = tags2id  # Convert bytes to int
        self.dumpdir = dumpdir
        self.gtfile = gtfile
        self.id_2_tracklet : dict[int, Tracklet] = {}

    def _generate_tracklets(self):
        # Read from cv_trace.csv and add the tracklets to id_2_tracklet
        gt_csv_path = os.path.join(self.dumpdir, "cv_trace.csv")

        cur_frame = -1
        with open(gt_csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in tqdm(reader):
                frame = int(row[0])

                if frame < cur_frame:
                    raise ValueError("Frame number is not in order")

                objid = int(row[1])
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])

                if objid not in self.id_2_tracklet:
                    self.id_2_tracklet[objid] = Tracklet(objid)

                self.id_2_tracklet[objid].add_frame(frame, x, y, w, h)

    def _assign_id_to_tracklets(self):
        # Read from matched_id.csv and gc.csv, and assign the id to the tracklets
        match_csv_path = os.path.join(self.dumpdir, "matched_id.csv")
        gc_csv_path = os.path.join(self.dumpdir, "gc.csv")

        with open(match_csv_path, 'r') as match_f, open(gc_csv_path, 'r') as gc_f:
            match_reader = csv.reader(match_f)
            gc_reader = csv.reader(gc_f)

            cur_tag_2_trackletid = {} # tagid -> trackletid

            match_end = False
            gc_end = False

            try:
                cur_match = next(match_reader)
                cur_match_f, cur_match_t, cur_match_id = int(cur_match[0]), self.tags2id[bytes.fromhex(cur_match[1])], int(cur_match[2])
            except StopIteration:
                match_end = True

            try:
                cur_gc = next(gc_reader)
                cur_gc_f, cur_gc_t = int(cur_gc[0]), self.tags2id[bytes.fromhex(cur_gc[1])]
            except StopIteration:
                gc_end = True

            while not match_end or not gc_end:
                # print("Cur match frame: ", cur_match_f, "Cur gc frame: ", cur_gc_f)
                if not match_end and (gc_end or cur_match_f < cur_gc_f):
                    # Add the matched point to the tracklet
                    cur_tag_2_trackletid[cur_match_t] = cur_match_id
                    self.id_2_tracklet[cur_match_id].add_matched_point(cur_match_f, cur_match_t)

                    try:
                        cur_match = next(match_reader)
                        cur_match_f, cur_match_t, cur_match_id = int(cur_match[0]), self.tags2id[bytes.fromhex(cur_match[1])], int(cur_match[2])
                    except StopIteration:
                        match_end = True
                        # print("Match end")
                else: # cur_match_f >= cur_gc_f or match_end
                    # Add the gc point to the tracklet
                    if cur_gc_t in cur_tag_2_trackletid:
                        self.id_2_tracklet[cur_tag_2_trackletid[cur_gc_t]].add_gc_point(cur_gc_f)
                        del cur_tag_2_trackletid[cur_gc_t]
                    else:
                        # print("No matched point for gc point")
                        pass

                    try:
                        cur_gc = next(gc_reader)
                        cur_gc_f, cur_gc_t = int(cur_gc[0]), self.tags2id[bytes.fromhex(cur_gc[1])]
                    except StopIteration:
                        gc_end = True
                        # print("GC end")

    def _back_identification(self):
        # Read from bc.csv and assign the id to the tracklets
        bc_csv_path = os.path.join(self.dumpdir, "bc.csv")
    
        with open(bc_csv_path, 'r') as bc_f:
            bc_reader = csv.reader(bc_f)

            for row in tqdm(bc_reader):
                tagid = self.tags2id[bytes.fromhex(row[1])]
                trackletid = int(row[2])

                self.id_2_tracklet[trackletid].add_bc_point(tagid)

    def _generate_gt(self, gen_non_identities=False):
        # Generate the gt.txt file from id_2_tracklet and cv_trace.csv, write to gtfile
        for key, tracklet in self.id_2_tracklet.items():
            tracklet.generate_assignments()

        cv_trace_path = os.path.join(self.dumpdir, "cv_trace.csv")
        with open(self.gtfile, 'w') as write_f, open(cv_trace_path, 'r') as read_f:
            reader = csv.reader(read_f)
            for row in tqdm(reader):
                frame = int(row[0])
                objid = int(row[1])
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])

                if objid in self.id_2_tracklet:
                    tracklet = self.id_2_tracklet[objid]
                    tagid = tracklet.get_identity(frame)
                    if tagid is not None:
                        write_f.write(f"{frame},{tagid},{x},{y},{w},{h},1,1,1\n")
                    else:
                        if gen_non_identities:
                            write_f.write(f"{frame},-1,{x},{y},{w},{h},1,1,1\n")

    def generate(self, gen_non_identities=False, back_identification=True):
        print("Generating tracklets")
        self._generate_tracklets()
        print("Assigning id to tracklets")
        self._assign_id_to_tracklets()
        if back_identification:
            print("Back identification")
            self._back_identification()
        else:
            print("Pass back identification")
        print("Generating gt")
        self._generate_gt(gen_non_identities)

class InterpolationGtFile:
    def __init__(self, gtpath, outpath, stack_size=5) -> None:
        self._stack_size = stack_size
        
        self.gtpath = gtpath
        self.outpath = outpath

    def interpolate(self):
        gt = pd.read_csv(self.gtpath, header=None)
        gt.columns = ["FrameID", "ObjectID", "X", "Y", "W", "H", "_", "__", "___"]
        gt["FrameID"] = gt["FrameID"].astype(int)
        gt["ObjectID"] = gt["ObjectID"].astype(int)

        # Group by object id
        grouped = gt.groupby("ObjectID")

        # Fill the frame gaps with nan if the gap is less than stack size
        # Then interpolate the nan values
        new_gts = []
        for objid, group in grouped:
            group = group.sort_values("FrameID")
            group = group.drop_duplicates(subset="FrameID")  # 删除重复的 FrameID

            group = group.set_index("FrameID")
            group = group.reindex(range(group.index.min(), group.index.max() + 1))
            group = group.interpolate(method='linear', limit=self._stack_size, limit_direction='both')
            
            # Remove the rows with nan values
            group = group.dropna()

            new_gts.append(group)

        new_gt = pd.concat(new_gts)
        new_gt["ObjectID"] = new_gt["ObjectID"].astype(int)

        # Sort the frame id
        new_gt = new_gt.sort_values("FrameID")

        new_gt.reset_index(inplace=True)

        new_gt.to_csv(self.outpath, header=False, index=False)
