import os
from data.KTH.kth_actions_frames import kth_actions_dict, settings, actions, person_ids

def convert_with_official_split():
    """
    len train 760
    len valid 768
    len test  863
    min_length train is 30
    min_length valid is 26
    min_length test is 24
    """
    for data_split in ['train', 'valid', 'test']:
        print('Converting ' + data_split)

        with open(f"./data/KTH/{data_split}-official.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        for frame_idxs in kth_actions_dict['person'+person_id][action][setting]:
                            file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                            file_path = os.path.join(action, file_name)
                            # index of kth_actions_frames.py starts from 1 but we need 0
                            # and wo should fix to [a,b) not [a,b]
                            # eg: 1-123, 124-345, length is 345
                            # ->  0-122, 123-344, length is 345
                            # ->  0-123, 123-345, length is 345 same
                            start_frame_idxs = frame_idxs[0] - 1
                            end_frames_idxs = frame_idxs[1]

                            min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                            f.write(f"{file_path} {start_frame_idxs} {end_frames_idxs}\n")
        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')

def convert_with_all_frames():
    """
    len train 191
    len valid 192
    len test  216
    min_length train is 250
    min_length valid is 204
    min_length test is 256
    """
    for data_split in ['train', 'valid', 'test']:
        print('Converting ' + data_split)

        with open(f"./data/KTH/{data_split}.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        a_list = sorted(kth_actions_dict['person'+person_id][action][setting])
                        # index of kth_actions_frames.py starts from 1 but we need 0
                        # and wo should fix to [a,b) not [a,b]
                        # eg: 1-12, ... 124-345, length is 345
                        # ->  0-11, ... 123-344, length is 345
                        # ->  0-345, length is 345 same
                        start_frame_idxs = a_list[0][0] - 1
                        end_frames_idxs = a_list[-1][1]

                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        file_path = os.path.join(action, file_name)
                        min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                        f.write(f"{file_path}\n")

        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')
        print("")

def convert_with_2_splits():
    """
    len train 382
    len valid 384
    len test  432
    min_length train is 105
    min_length valid is 98
    min_length test is 84
    """
    for data_split in ['train', 'valid', 'test']:
        print('Converting ' + data_split)

        with open(f"./data/KTH/{data_split}-2splits.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        # index of kth_actions_frames.py starts from 1 but we need 0
                        # and wo should fix to [a,b) not [a,b]
                        # eg: 1-12, ... 124-345, length is 345
                        # ->  0-11, ... 123-344, length is 345
                        # ->  0-345, length is 345 same

                        a_list = sorted(kth_actions_dict['person'+person_id][action][setting])

                        start_frame_idxs = a_list[0][0] - 1
                        end_frames_idxs = a_list[1][1]

                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        file_path = os.path.join(action, file_name)
                        min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                        f.write(f"{file_path} {start_frame_idxs} {end_frames_idxs}\n")

                        start_frame_idxs = a_list[-2][0] - 1
                        end_frames_idxs = a_list[-1][1]

                        file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                        file_path = os.path.join(action, file_name)
                        min_length = min(min_length, end_frames_idxs - start_frame_idxs)
                            
                        f.write(f"{file_path} {start_frame_idxs} {end_frames_idxs}\n")
                        
        print('min_length', data_split, 'is', min_length)
        print('Converting', data_split, 'done.')
        print("")

def convert_into_mini_splits(train, valid, test):
    """
    len train 25535
    len valid 256
    len test  256
    min_length train is 15
    min_length valid is 20
    min_length test is 50
    """
    for data_split in ['train', 'valid', 'test']:

        count = 0

        if data_split == "train":
            frame_num = train
        elif data_split == "valid":
            frame_num = valid
        else:
            frame_num = test
        
        print('Converting ' + data_split)

        with open(f"./data/KTH/{data_split}-mini.txt", 'w') as f:
            min_length = 1e6
            split_person_ids = person_ids[data_split]
            for person_id in split_person_ids:
                # print('     Converting person' + person_id)
                for action in kth_actions_dict['person'+person_id]:
                    for setting in kth_actions_dict['person'+person_id][action]:
                        for setting in kth_actions_dict['person'+person_id][action]:

                            if data_split == "train":
                                
                                a_list = sorted(kth_actions_dict['person'+person_id][action][setting])
                                start_frame_idx = a_list[0][0] - 1
                                end_frames_idx = a_list[-1][1]
                                get_clip_num = (end_frames_idx - start_frame_idx) //  frame_num
                                for i in range(get_clip_num):
                                    file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                                    file_path = os.path.join(action, file_name)         
                                    f.write(f"{file_path} {i*frame_num} {(i+1)*frame_num}\n")
                                    count += 1

                            else:
                                for frame_idxs in kth_actions_dict['person'+person_id][action][setting]:
                                    file_name = 'person' + person_id + '_' + action + '_' + setting + '_uncomp.avi'
                                    file_path = os.path.join(action, file_name)
                                    start_frame_idxs = frame_idxs[0] - 1
                                    end_frames_idxs = frame_idxs[1]

                                    if frame_num <= end_frames_idxs - start_frame_idxs:
                                        f.write(f"{file_path} {start_frame_idxs} {start_frame_idxs + frame_num}\n")
                                        count += 1
                                        break
                                
        # make_valid_and_test_as_max256 batches for calculate FVD anb other metrics 
        if data_split != "train" and count > 256:
            import random
            random.seed(2023)     
            with open(f"./data/KTH/{data_split}-mini.txt", 'r') as f:
                lines = f.readlines()
            lines = random.sample(lines, 256)
            count = 256
            with open(f"./data/KTH/{data_split}-mini.txt", 'w') as f:
                f.writelines(lines)
        print(f"count is {count}")
    print("")

# convert_with_official_split()
# convert_with_all_frames()
# convert_with_2_splits()
convert_into_mini_splits(15,20,50)