from sympy import sympify


def push(obj, l, depth):
    """Helper function for parsing a string into XADD"""
    while depth:
        l = l[-1]
        depth -= 1

    l.append(obj)


def parse_xadd_grammar(s, ns):
    """Helper function for parsing a string into XADD"""
    groups = []
    depth = 0
    sympyLst = []
    s = s.strip()

    try:
        i = 0
        while i < len(s):
            if s[i] == '(':
                push([], groups, depth)
                push([], sympyLst, depth)
                depth += 1
                i += 1
            elif s[i] == ')':
                depth -= 1
                i += 1
            else:
                idx_next_open = s.find('(', i, -1)
                idx_next_close = s.find(')', i)
                if idx_next_open < 0 and idx_next_close < 0:
                    break
                elif idx_next_open < 0 and idx_next_close > 0:
                    idx_next_paren = idx_next_close
                elif idx_next_open > 0 and idx_next_close < 0:
                    raise ValueError
                elif idx_next_open > 0 and idx_next_close > 0:
                    idx_next_paren = min(idx_next_open, idx_next_close)

                s_chunk = s[i: idx_next_paren]
                i += len(s_chunk)
                s_chunk = s_chunk.strip()
                if s_chunk:
                    push(s_chunk, groups, depth)
                    push(sympify(s_chunk.strip('([])'), locals=ns), sympyLst, depth)

    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups, sympyLst
