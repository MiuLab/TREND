import ipdb

def cal_seen_unseen_stats(preds, gts):

    unseen = set([5, 11, 12])

    unseen_total = unseen_match = 0
    seen_total = seen_match = 0
    for i in range(len(gts)):
        if gts[i] in unseen:
            unseen_total += 1
            if preds[i] == gts[i]:
                unseen_match += 1
        else:
            seen_total += 1
            if preds[i] == gts[i]:
                seen_match += 1

    print(f"seen_total: {seen_total}")
    print(f"seen_match: {seen_match}")
    print(seen_match / seen_total)
    print(f"unseen_total: {unseen_total}")
    print(f"unseen_match: {unseen_match}")
    print(unseen_match / unseen_total)


