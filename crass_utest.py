import crass
from pyannotate_runtime import collect_types

if __name__ == '__main__':
    collect_types.init_types_collection()
    collect_types.resume()
    crass()
    collect_types.pause()
collect_types.dump_stats('type_info.json')