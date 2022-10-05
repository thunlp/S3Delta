


class Debug:
    count_dict = {}
    def __init__(self, place="", crit=lambda x:x>50):
        if place not in Debug.count_dict:
            Debug.count_dict[place] = 0
        else:
            Debug.count_dict[place] += 1
        self.count = Debug.count_dict[place]
        if crit(self.count):
            self.embed = True
        else:
            self.embed = False

# def get_embed(header="", crit=lambda x:x>50):
#     _heading_ = header
#     _debug_ = Debug(_heading_, lambda x:x>50)
#     if _debug_.embed:
#         from IPython import embed
#         embed(header = f"{_heading_}_{_debug_.count}")