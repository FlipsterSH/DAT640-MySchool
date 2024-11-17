from typing import List, Union


def get_unique_elements(
    lst: List[Union[str, int]], n: int = 1
) -> List[Union[str, int]]:
    """Given a list of elements returns those that repeat at least n times. The
    output list should contain all unique elements and they should be returned
    in the same order as they first appear in the input list.

    Args:
        lst: Input list
        n (optional): Minimum number of times an element should be repeated to
            be returned. Defaults to 1.

    Returns:
        List of unique items
    """
    try:
        elements = {}
        return_list = []
        
        if len(lst) > 0:
            for e in lst:
                if e not in elements:
                    elements[e] = 1
                else:
                    elements[e] += 1
            
            print(elements)
            for r in elements:
                if elements[r] >= n:
                    return_list.append(r)

        return return_list

    except:
        return lst