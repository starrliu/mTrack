import json
from enum import Enum

import zmq
from bitarray import bitarray


class StateUnawareAction(Enum):
    SELECT = 0
    DESELECT = 1
    DONOTHING = 2


class SelectCommand:
    def __init__(
        self,
        tagmask: str,
        bitpointer: int,
        bitcount: int,
        matchaction: StateUnawareAction,
        nonmatchaction: StateUnawareAction,
    ):
        self.tagmask: str = tagmask  # "f" or "0"
        self.bitpointer: int = bitpointer
        self.bitcount: int = bitcount
        self.matchaction: StateUnawareAction = matchaction
        self.nonmatchaction: StateUnawareAction = nonmatchaction

    def to_dict(self) -> dict:
        return {
            "tagmask": self.tagmask,
            "bitpointer": self.bitpointer,
            "bitcount": self.bitcount,
            "matchaction": self.matchaction.value,
            "nonmatchaction": self.nonmatchaction.value,
        }

    def __str__(self) -> str:
        return f"SelectCommand(tagmask={self.tagmask}, bitpointer={self.bitpointer}, bitcount={self.bitcount}, matchaction={self.matchaction.name}, nonmatchaction={self.nonmatchaction.name})"


class BatchSelect:
    def __init__(
        self, epcid_bit_length: int, epcids: list[bytes], max_select_count: int = 5
    ):
        self.epcid_bit_length: int = epcid_bit_length
        # Check the length of epcids
        for epcid in epcids:
            if len(epcid) * 8 != epcid_bit_length:
                raise ValueError("EPCID length is not matched")
        self.epcids_set: set[bytes] = set(epcids)

        """
        Mask pool is a dictionary that indicates the tags which has 1 in the specific bit position.
        
        self.mask_pool[bit_position] = {tag1, tag2, tag3, ...}
        indicates that tag1, tag2, tag3, ... has 1 in the bit_position.
        """
        self.mask_pool: dict[tuple[int, int], set[bytes]] = (
            dict()
        )  # key: (bit_position, one or zero), value: set of tags
        self.__gen_mask_pool()

        self.max_select_count: int = max_select_count

    def __gen_mask_pool(self):
        """
        Generate mask pool from epcids_set.
        """

        for idx in range(self.epcid_bit_length):
            zero_tags = set()
            one_tags = set()
            for epcid in self.epcids_set:
                epcid_bitarray = bitarray()
                epcid_bitarray.frombytes(epcid)
                if epcid_bitarray[idx]:
                    one_tags.add(epcid)
                else:
                    zero_tags.add(epcid)

            if len(one_tags) > 0:
                self.mask_pool[(idx, 1)] = one_tags
            if len(zero_tags) > 0:
                self.mask_pool[(idx, 0)] = zero_tags

    def select_batch(
        self, subset_tags: set[bytes]
    ) -> tuple[list[SelectCommand], set[bytes]]:
        """
        Select tags in the subset_tags.

        Algorithm:
        1. Try to select the tags as many as possible.
            1.1 Sort the idx of mask_pool by the size of intersection of
                subset_tags and mask_pool[idx].
            1.2 Select the idx which has the largest intersection.
                If there are multiple idxs, select the idx which has the
                smallest size of mask_pool[idx].
        2. De-select the tags which are not in the subset_tags.
            2.1 Calculate the tags which are not in the subset_tags but are selected.
            2.2 Find all the idxs in the mask_pool which only have the tags in the calculated tags.
            2.3 De-select the tags in the idxs.
        """

        selectcmds = []

        # Step 1: Select the tags as many as possible
        first_cmd = True
        unselected_tags = subset_tags.copy()
        selected_tags = set()
        while len(unselected_tags) > 0:
            idxs = list(self.mask_pool.keys())
            idxs.sort(
                key=lambda x: (
                    len(self.mask_pool[x] & unselected_tags),
                    -len(self.mask_pool[x]),
                ),
                reverse=True,
            )

            if len(idxs) == 0:
                break

            selected_idx = idxs[0][0]
            selected_value = idxs[0][1]

            cur_tags = self.mask_pool[idxs[0]]

            selected_tags |= cur_tags
            unselected_tags -= cur_tags

            # Generate SelectCommand
            if selected_value == 1:
                tagmask = "f"
            else:
                tagmask = "0"

            bitpointer = selected_idx
            bitcount = 1
            matchaction = StateUnawareAction.SELECT
            if first_cmd:
                nonmatchaction = StateUnawareAction.DESELECT
            else:
                nonmatchaction = StateUnawareAction.DONOTHING

            selectcmd = SelectCommand(
                tagmask, bitpointer, bitcount, matchaction, nonmatchaction
            )
            selectcmds.append(selectcmd)

            if len(selectcmds) == self.max_select_count:
                print("Select count is reached")
                return selectcmds, selected_tags

        # Step 2: De-select the tags which are not in the subset_tags
        to_deselect_tags = selected_tags - subset_tags
        while len(to_deselect_tags) > 0:
            deselect_mask_idxs = [
                idx
                for idx in self.mask_pool.keys()
                if self.mask_pool[idx].issubset(to_deselect_tags)
            ]
            # Sort the idxs by the size of mask_pool[idx]
            deselect_mask_idxs.sort(key=lambda x: len(self.mask_pool[x]), reverse=True)

            if len(deselect_mask_idxs) == 0:
                return selectcmds, selected_tags

            deselect_idx = deselect_mask_idxs[0][0]
            deselect_value = deselect_mask_idxs[0][1]

            cur_tags = self.mask_pool[deselect_mask_idxs[0]]

            to_deselect_tags -= cur_tags
            selected_tags -= cur_tags

            # Generate SelectCommand
            # tagmask = bitarray([deselect_value])
            if deselect_value == 1:
                tagmask = "f"
            else:
                tagmask = "0"
            bitpointer = deselect_idx
            bitcount = 1
            matchaction = StateUnawareAction.DESELECT
            nonmatchaction = StateUnawareAction.DONOTHING
            selectcmd = SelectCommand(
                tagmask, bitpointer, bitcount, matchaction, nonmatchaction
            )
            selectcmds.append(selectcmd)

            if len(selectcmds) == self.max_select_count:
                print("Select count is reached")
                return selectcmds, selected_tags

        return selectcmds, selected_tags


class FilterManager:
    """
    FilterManager is a class that manages the filters for the tags.
    """

    def __init__(self, tags: list[bytes], address="tcp://*:5556") -> None:
        self.filters = set()
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(address)

        self.batch_select = BatchSelect(len(tags[0]) * 8, tags)

    def set_filters(self, tags: set[bytes]) -> None:
        """
        Set the filters of the FilterManager.

        Args:
            tags(set[bytes]): the tags to be set.
        """
        self.filters = tags

    def add_filter(self, tag_id: bytes) -> None:
        """
        Add a tag filter to the FilterManager.

        Args:
            tag_id(bytes): the tag id to be added.
        """
        if tag_id in self.filters:
            return

        self.filters.add(tag_id)

    def remove_filter(self, tag_id: bytes) -> None:
        if tag_id not in self.filters:
            return

        self.filters.remove(tag_id)

    def send(self) -> tuple[list[SelectCommand], set[bytes]]:
        """
        Send the select commands to the RFID reader.
        """

        selectcmds, selected_tags = self.batch_select.select_batch(self.filters)
        dict_list = [selectcmd.to_dict() for selectcmd in selectcmds]

        json_data = json.dumps(dict_list)

        self.socket.send_string(json_data)

        return selectcmds, selected_tags

    def close(self) -> None:
        """
        Close the FilterManager.
        """
        self.socket.close()
        self.context.term()

    def __len__(self):
        return len(self.filters)

    def __del__(self):
        self.close()
