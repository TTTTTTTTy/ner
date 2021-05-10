
class Tagger:
    def __init__(self, tag_list, pad_tag="<PAD>"):
        self.PAD_TAG = pad_tag
        self.tag_list = tag_list
        self.id_to_tag = {i:tag for i, tag in enumerate(tag_list)}
        self.id_to_tag[-1] = self.PAD_TAG
        self.tag_to_id = {tag:i for i, tag in enumerate(tag_list)}
        self.tag_to_id[self.PAD_TAG] = -1
    
    '''
    Input:one tag lists, like[B, I, I, O, B, O]
    '''
    def convert_tags_to_ids(self, tags):  # one tag list
        return [self.tag_to_id[tag] for tag in tags]
    
    '''
    Input: list of tag lists, like [ [B, I, I, O, B, O], [O, B, I, O] ]
    '''
    def convert_batch_tags_to_ids(self, tags_lst, padding=False, max_length=None):  
        ids_lst = []
        for tags in tags_lst:
            ids_lst.append(self.convert_tags_to_ids(tags))
        lengths = None
        if padding:
            lengths = [ len(tags) for tags in tags_lst]
            # if user don't set max_length , padding to the length of longest sentences
            if max_length == None:     
                max_length = max(lengths)   
            for ids in ids_lst:
                ids.extend([self.tag_to_id[self.PAD_TAG]] * (max_length - len(ids))) # pad to max length
        return ids_lst, lengths
    
    def convert_ids_to_tags(self, ids):
        return [self.id_to_tag[ix] for ix in ids if self.id_to_tag[ix] != self.PAD_TAG]
    
    def convert_batch_ids_to_tags(self, ids_lst):
        return [ self.convert_ids_to_tags(ids) for ids in ids_lst]  
    