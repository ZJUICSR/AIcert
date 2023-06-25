def make_term(freeze_bn, reuse_statistic, reuse_teacher_statistic):
    if isinstance(freeze_bn, bool):
        term = str(freeze_bn)
    else:
        if isinstance(freeze_bn, tuple):
            term = f"b{freeze_bn[0]}t{freeze_bn[1]}"
        elif isinstance(freeze_bn, list):
            min_block = min(freeze_bn)
            max_block = max(freeze_bn)
            term = f"b{min_block}t{max_block}"
        else:
            raise ValueError(f"provided 'freeze_bn' is {type(freeze_bn)}, but only [bool|tuple|list] is supported")
        
    if reuse_statistic:
        term += "_rs" # rs: reuse_statistic
    if reuse_teacher_statistic:
        term = "rts_" + term # rts: reuse teacher statistic
    
    # [some valid returned pattern]:
    # False
    # True
    #
    # rts_True
    # True_rs
    # rts_True_rs
    #
    # b*t* 
    # rts_b*t*
    # rts_b*t*_rs
    # b*t*_rs

    return term

if __name__ == '__main__':
    import argparse
    class FreezeBNAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if len(values) == 0:
                freeze_bn = True
            else:
                if len(values) == 1:
                    import re
                    if re.match(r"\((\d+),(\d+)\)", values[0]) is None:
                        raise ValueError("Assume you provide a tuple. please pass it in '(min, max)'.")
                    [_, min_, max_, _] = re.split(r"\((\d+),(\d+)\)", values[0])
                    freeze_bn = (int(min_), int(max_))
                else:
                    freeze_bn = list(map(lambda x: int(x), values))

            setattr(namespace, self.dest, freeze_bn)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-bn", action=FreezeBNAction, nargs="*", default=False)
    parser.add_argument("--reuse-statistic", action="store_true")
    parser.add_argument("--reuse-teacher-statistic", action="store_true")

    args = parser.parse_args()
    print(make_term(args.freeze_bn, args.reuse_statistic, reuse_teacher_statistic=args.reuse_teacher_statistic))