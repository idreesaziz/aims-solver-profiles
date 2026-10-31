"""Human-written question features and bounded exact interpreters; text only.

No item IDs, benchmark fields, reference answers, learned labels or network calls
are inputs. An exact-result feature is explicitly distinguished from a demand.
"""
import ast
from functools import lru_cache
import itertools
import re
import numpy as np


def stem_options(text):
    chunks=re.split(r'\bOptions:\s*',text)
    stem=chunks[0].strip();choices=[]
    for block in chunks[1:]:
        alpha=re.findall(r'^\([A-Z]\)\s*(.+)$',block,re.M)
        numeric=re.findall(r'^\d+\.\s*(.+)$',block,re.M)
        bullet=re.findall(r'^-\s*(.+)$',block,re.M)
        if alpha:choices=alpha;break
        if bullet:choices=bullet;break
        if numeric and not all(re.fullmatch(r'\([A-Z]\)',v.strip()) for v in numeric):choices=numeric;break
        if block.lstrip().startswith('['):
            try:
                values=ast.literal_eval(block.strip())
                if isinstance(values,list) and all(isinstance(v,str) for v in values):choices=values;break
            except (ValueError,SyntaxError):pass
    return stem,[v.strip() for v in choices]


def normalize(text):return ' '.join(text.casefold().strip().rstrip('.!?').split())


def boolean_value(stem):
    if not re.fullmatch(r'[\s()TrueFalsnotadorexi]+',stem):return None
    expression=re.sub(r'\s+is\s*$', '',stem).strip()
    if not re.fullmatch(r'(?:True|False|not|and|or|\(|\)|\s)+',expression):return None
    def visit(node):
        if isinstance(node,ast.Expression):return visit(node.body)
        if isinstance(node,ast.Constant) and type(node.value) is bool:return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not):return not visit(node.operand)
        if isinstance(node,ast.BoolOp) and isinstance(node.op,(ast.And,ast.Or)):
            values=[visit(v) for v in node.values]
            return all(values) if isinstance(node.op,ast.And) else any(values)
        raise ValueError('Only Boolean literals/operators allowed')
    try:return visit(ast.parse(expression,mode='eval'))
    except (ValueError,SyntaxError,RecursionError):return None


def truth_chain(stem):
    text=re.sub(r'^Question:\s*','',stem)
    start=re.match(r'(\w+) (tells the truth|lies)\.\s*',text)
    if not start:return None
    query=re.search(r'Does (\w+) tell the truth\?\s*$',text)
    if not query:return None
    middle=text[start.end():query.start()];person=start[1];value=start[2]=='tells the truth';end=0;steps=0
    for match in re.finditer(r'(\w+) says (\w+) (tells the truth|lies)\.\s*',middle):
        if match.start()!=end:return None
        if match[2]!=person:return None
        person=match[1];value=value if match[3]=='tells the truth' else not value;steps+=1;end=match.end()
    return value if steps and end==len(middle) and query[1]==person else None


def navigation(stem):
    prefix='If you follow these instructions, do you return to the starting point?'
    if not stem.startswith(prefix):return None
    text=stem[len(prefix):].strip();absolute=False
    if text.startswith('Always face forward.'):
        absolute=True;text=text[len('Always face forward.'):].strip()
    x=y=direction=0
    for clause in text.split('.'):
        clause=clause.strip()
        if not clause:continue
        if clause in ['Turn left','Turn right','Turn around']:
            if absolute:return None
            direction=(direction+{'Turn left':-1,'Turn right':1,'Turn around':2}[clause])%4;continue
        match=re.fullmatch(r'Take (\d+) steps?(?: (forward|backward|left|right))?',clause)
        if not match:return None
        n=int(match[1]);word=match[2]
        if absolute and word is None:return None
        if not absolute and word is not None:return None
        dx,dy=({'forward':(0,1),'backward':(0,-1),'left':(-1,0),'right':(1,0)}[word] if absolute else [(0,1),(1,0),(0,-1),(-1,0)][direction])
        x+=n*dx;y+=n*dy
    return dict(returns=x==y==0,axis_cancel=(x==0 or y==0),absolute=absolute)


def swaps(stem,options):
    clauses=re.findall(r'\b(?:First|Then|Finally),\s*([^.!?]+)[.!?]',stem)
    if not clauses or not re.search(r'swap|switch partners|trade positions', ' '.join(clauses)):return None
    pairs=[]
    for clause in clauses:
        match=re.fullmatch(r'(\w+) and (\w+) (?:swap|switch|trade) .+',clause)
        if not match:return None
        pairs.append((match[1],match[2]))
    query=re.search(r'At the end of[^.!?]*?,\s*(\w+)\s+(?:has|is|gets)',stem)
    if not query:return None
    target=query[1];owner=target;depth=0
    for a,b in reversed(pairs):
        if owner==a:owner=b;depth+=1
        elif owner==b:owner=a;depth+=1
    # Explicit starting roster, before the first . All option slots align with
    # the displayed initial owners in this format, verified separately below.
    initial=stem.split('First,',1)[0]
    roster=[]
    for match in re.finditer(r'\b([A-Z][a-z]+) (?:has|is (?:playing|dancing)|gets)\b',initial):
        if match[1] not in roster:roster.append(match[1])
    choice=roster.index(owner) if owner in roster and len(roster)==len(options) else None
    return dict(depth=depth,restored=owner==target,choice=choice,total=len(pairs),roster=roster)


NUMBERS=dict(zero=0,one=1,two=2,three=3,four=4,five=5,six=6,seven=7,eight=8,nine=9,ten=10,a=1,an=1)
CATEGORIES={
    'animals':set('bear cat chicken cow dog donkey duck fish frog goat mouse pig rabbit snail snake'.split()),
    'fruits':set('apple banana blackberry grape nectarine orange peach plum raspberry strawberry'.split()),
    'vegetables':{'cabbage','carrot','cauliflower','garlic','head of broccoli','lettuce head','onion','potato','stalk of celery','yam'},
    'musical instruments':set('accordion clarinet drum flute piano trombone trumpet violin'.split()),
    'objects':set('bed car chair couch fridge lamp microwave oven stove table toaster'.split())}
PLURAL={'mice':'mouse','couches':'couch','potatoes':'potato','peaches':'peach','blackberries':'blackberry','raspberries':'raspberry','strawberries':'strawberry','heads of broccoli':'head of broccoli','stalks of celery':'stalk of celery','lettuce heads':'lettuce head'}
ALL_NOUNS=set.union(*CATEGORIES.values())


def inventory(stem):
    match=re.fullmatch(r'I have (.+)\. How many (.+?) do I have\?',stem,re.S)
    if not match or match[2] not in CATEGORIES:return None
    items=re.split(r',\s*(?:and\s+)?|\s+and\s+',match[1]);quantities=[];filtered=False;answer=0
    for item in items:
        fields=item.strip().split(' ',1)
        if len(fields)!=2 or fields[0] not in NUMBERS:return None
        n=NUMBERS[fields[0]];noun=fields[1];noun=PLURAL.get(noun,noun[:-1] if noun.endswith('s') and noun[:-1] in ALL_NOUNS else noun)
        if noun not in ALL_NOUNS:return None
        quantities.append(n)
        if match[2]=='objects' or noun in CATEGORIES[match[2]]:answer+=n
        else:filtered=True
    return dict(total=answer,mixed=max(quantities)>1 and min(quantities)==1,plural=max(quantities)>1,filtered=filtered)


@lru_cache(maxsize=4)
def permutations(n):return np.asarray(list(itertools.permutations(range(n))),dtype=np.int8)


def rank_value(phrase,n):
    phrase=phrase.strip().strip('.').casefold().replace('-',' ')
    phrase=re.sub(r'\s+',' ',phrase)
    phrase=re.sub(r'^the ','',phrase)
    order=dict(first=1,second=2,third=3,fourth=4,fifth=5,sixth=6,seventh=7)
    if phrase in ['first','oldest','cheapest','leftmost']:return 0
    if phrase in ['last','newest','most expensive','rightmost']:return n-1
    if phrase in order:return order[phrase]-1
    for ordinal,value in order.items():
        if phrase in [ordinal+' oldest',ordinal+' cheapest',ordinal+' from the left']:return value-1
        if phrase in [ordinal+' newest',ordinal+' most expensive',ordinal+' from the right',ordinal+' to last']:return n-value
    return None


def ordering(stem,options):
    if 'arranged in a fixed order' not in stem:return None
    entities=[];queries=[]
    for option in options:
        match=re.fullmatch(r'(?:The )?(.+?) (?:is|are|finished) (.+)',option,re.I)
        if not match:return None
        entities.append(match[1].casefold());queries.append(match[2])
    n=len(entities)
    if n not in [3,5,7] or len(set(entities))!=n:return None
    indices={name:i for i,name in enumerate(entities)};pattern='|'.join(re.escape(name) for name in sorted(entities,key=len,reverse=True))
    # Anchored sentence statements avoid matching relation mentions in setup.
    facts=[]
    for statement in re.split(r'[.!?]\s*',stem):
        match=re.fullmatch(r'(?:The )?('+pattern+r') (?:is|are|finished) (.+)',statement.strip(),re.I)
        if not match:continue
        a=indices[match[1].casefold()];rest=match[2].casefold()
        relation=re.fullmatch(r'(older than|newer than|less expensive than|more expensive than|to the left of|to the right of|above|below) (?:the )?('+pattern+r')',rest,re.I)
        if relation:
            b=indices[relation[2].casefold()];less=relation[1] in ['older than','less expensive than','to the left of','above'];facts.append(('lt',a,b) if less else ('lt',b,a))
        else:
            rank=rank_value(rest,n)
            if rank is None:return None
            facts.append(('eq',a,rank))
    if not facts:return None
    p=permutations(n);mask=np.ones(len(p),bool)
    for kind,a,b in facts:mask&=p[:,a]<p[:,b] if kind=='lt' else p[:,a]==b
    p=p[mask]
    if not len(p):return None
    correct=[]
    for i,query in enumerate(queries):
        rank=rank_value(query,n)
        if rank is None:return None
        if np.all(p[:,i]==rank):correct.append(i)
    if len(correct)!=1:return None
    sentences={normalize(s) for s in re.split(r'[.!?]\s*',stem) if s.strip()}
    direct=any(normalize(o) in sentences for o in options)
    return dict(choice=correct[0],direct=direct,extreme=rank_value(queries[0],n) in [0,n-1])


# Each predicate is authored explicitly by Codex after inspecting source text.
FEATURES=[
 ('open_response','No discrete candidate answer options are supplied','format'),
 ('open_symbolic','Open response with symbolic mathematics','structure'),
 ('narrative_belief','Ask where a character would look for an object','structure'),
 ('narrative_allocate','Ask for allocation of people to tasks','structure'),
 ('narrative_murder','Ask for the most likely murderer','structure'),
 ('boolean_expression','Propositional expression of Boolean literals/operators','structure'),
 ('boolean_false','Parsed Boolean expression evaluates to False','exact_result'),
 ('truth_chain','Linked sequence of truth/lie assertions','structure'),
 ('truth_true','Validated truth/lie chain ends true','exact_result'),
 ('navigation','Instructions ask whether a walk returns to its origin','structure'),
 ('navigation_return','Validated walk returns to its origin','exact_result'),
 ('navigation_absolute','Navigation gives absolute forward/back/left/right steps','structure'),
 ('navigation_axis_cancel','At least one displacement axis cancels exactly','exact_result'),
 ('swaps','Sequential pairwise exchange of held items or roles','structure'),
 ('swap_depth_one','Backward dependency trace uses at most one exchange','exact_result'),
 ('swap_depth_two','Backward dependency trace uses exactly two exchanges','exact_result'),
 ('swap_restored','Queried entity receives its initial item/role again','exact_result'),
 ('ordering','Rank or linear-position deduction from inequalities','structure'),
 ('ordering_literal','A candidate statement is literally asserted in premise','structure'),
 ('ordering_extreme','Requested rank is an endpoint of the ordering','structure'),
 ('inventory','Explicit quantified inventory and category-total question','structure'),
 ('inventory_mixed','Inventory mixes plural quantities and singular items','structure'),
 ('inventory_filtered','Some inventory nouns are outside the requested category','structure'),
 ('inventory_over_six','Computed inventory total exceeds six','exact_result'),
 ('exact_first_choice','Validated discrete interpreter selects first option','exact_result'),
 ('exact_last_choice','Validated discrete interpreter selects last option','exact_result'),
 ('exact_later_half','Validated discrete interpreter selects later half of options','exact_result'),
 ('explicit_composite_output','Explicit pair/vector/interval/comma-separated output','structure'),
 ('matrix_input','Question supplies a mathematical matrix','structure'),
 ('recurrence','An equation relates adjacent sequence/function indices','structure'),
 ('complex_trig','Complex numbers, roots of unity or trigonometric functions','domain'),
 ('number_theory','Congruences, divisibility, gcd or lcm','domain'),
 ('geometry_diagram','Embedded Asymptote geometric construction','structure'),
 ('direct_function_substitution','Defined function followed by direct substitution request','structure'),
 ('remove_absolute_value','Rewrite an expression without absolute-value signs','structure'),
 ('rate_units','Motion, purchase or unit-rate arithmetic','domain'),
 ('math_universal_extremum','Universal/existence/extremum mathematical constraints','structure'),
 ('negative_request','Explicit negative or exception selection in requested question','structure'),
 ('duplicate_options','At least two actual displayed options have identical text','quality'),
 ('numeric_options','Every displayed choice is a scalar number','structure'),
 ('chemical_transformation','Question explicitly describes a chemical transformation','domain'),
 ('biomedical','Biomedical organisms, drugs, genes or physiology','domain'),
 ('physics_quantities','Numerical physical quantities and physical laws','domain'),
 ('geometric_svg','SVG path shape identification','structure'),
 ('svg_many_sides','Parsed SVG polygon has at least seven distinct vertices','exact_result'),
 ('table_query','Question asks about a displayed named table','structure'),
 ('table_modified','Displayed table has a deletion or insertion','structure'),
 ('table_filter','Question asks for a conjunction of table conditions','structure'),
 ('target_restoration_cue','Target-object sentence explicitly mentions a return/back','structure'),
 ('interpreter_supported','A bounded deterministic interpreter fully supports the stated discrete problem','coverage'),
 ('target_query_move_clause','Queried person is named immediately before a movement of queried object','structure'),
 ('allocation_solitary_cue','Allocation narrative explicitly mentions working alone/individually','structure')]


def extract(text):
    stem,options=stem_options(text);lower=stem.casefold();v={key:0 for key,_,_ in FEATURES};diagnostics={}
    symbolic='$' in stem or '\\[' in stem
    v['open_response']=int(not options)
    v['open_symbolic']=int(not options and symbolic)
    v['narrative_belief']=int('would look to find' in lower)
    v['narrative_allocate']=int('allocate each person' in lower or 'allocate the people' in lower)
    v['narrative_murder']=int('most likely murderer' in lower)
    b=boolean_value(stem);t=truth_chain(stem);nav=navigation(stem);sw=swaps(stem,options);order=ordering(stem,options);inv=inventory(stem)
    results=[]
    if b is not None:
        v['boolean_expression']=1;v['boolean_false']=int(not b)
        if str(b) in options:results.append(('boolean',options.index(str(b))))
    if t is not None:
        v['truth_chain']=1;v['truth_true']=int(t)
        answer='yes' if t else 'no'
        if answer in [o.casefold() for o in options]:results.append(('truth_chain',[o.casefold() for o in options].index(answer)))
    if nav is not None:
        v.update(navigation=1,navigation_return=int(nav['returns']),navigation_absolute=int(nav['absolute']),navigation_axis_cancel=int(nav['axis_cancel']))
        answer='yes' if nav['returns'] else 'no'
        if answer in [o.casefold() for o in options]:results.append(('navigation',[o.casefold() for o in options].index(answer)))
    if sw is not None:
        v.update(swaps=1,swap_depth_one=int(sw['depth']<=1),swap_depth_two=int(sw['depth']==2),swap_restored=int(sw['restored']))
        diagnostics['swaps']=sw
        if sw['choice'] is not None:results.append(('swaps',sw['choice']))
    if order is not None:
        v.update(ordering=1,ordering_literal=int(order['direct']),ordering_extreme=int(order['extreme']));results.append(('ordering',order['choice']))
    elif 'arranged in a fixed order' in lower:
        v['ordering']=1
        sentences={normalize(s) for s in re.split(r'[.!?]\s*',stem) if s.strip()};v['ordering_literal']=int(any(normalize(o) in sentences for o in options))
    if inv is not None:
        v.update(inventory=1,inventory_mixed=int(inv['mixed']),inventory_filtered=int(inv['filtered']),inventory_over_six=int(inv['total']>6))
        if str(inv['total']) in options:results.append(('inventory',options.index(str(inv['total']))))
    if results:
        assert len(results)==1,'Ambiguous interpreter scopes'
        kind,choice=results[0];v.update(exact_first_choice=int(choice==0),exact_last_choice=int(choice==len(options)-1),exact_later_half=int(choice>=len(options)/2));diagnostics['exact_result']=dict(kind=kind,choice=choice,options=len(options))
    v['interpreter_supported']=int(bool(results))
    v['explicit_composite_output']=int(bool(re.search(r'(?:answer|enter|express|find).{0,100}(?:ordered pair|interval notation|separated by commas|column vector)',lower,re.S)))
    v['matrix_input']=int('\\begin{pmatrix}' in stem or '\\begin{bmatrix}' in stem)
    v['recurrence']=int(bool(re.search(r'[fa]\s*\(\s*n\s*[+-]\s*1\s*\)',stem)))
    v['complex_trig']=int(bool(re.search(r'complex number|root.{0,10}unity|\\(?:sin|cos|tan|omega)\b',stem,re.I)))
    v['number_theory']=int(bool(re.search(r'\\pmod|divisib|greatest common divisor|least common multiple',stem,re.I)))
    v['geometry_diagram']=int('[asy]' in stem)
    v['direct_function_substitution']=int(bool(re.search(r'(?:defined as|defined by|let).{0,45}f\(x\)\s*=',stem,re.I|re.S) and re.search(r'(?:find|describe|compute|value of).{0,35}f\([^)]*\)',stem,re.I|re.S)))
    v['remove_absolute_value']=int('without' in lower and 'absolute value' in lower)
    v['rate_units']=int(bool(re.search(r'miles|kilometers|kilometres|inches per second|dollars a pound|at this rate',lower)))
    v['math_universal_extremum']=int(symbolic and bool(re.search(r'for all|there exist|maximum|minimum|least possible|greatest possible',lower)))
    requests=list(re.finditer(r'(?:^|[.!?]\s+|\n)\s*(?:which|what|how|who|where|when|did|does|can|could|will|would|is|are|should|find|compute|determine|identify|all of the following)\b',lower))
    request=lower[requests[-1].start():] if requests else ''
    v['negative_request']=int(bool(re.search(r'\b(?:not|cannot|false|incorrect|except|neither|unlikely)\b',request)))
    normalized=[normalize(o) for o in options];v['duplicate_options']=int(len(normalized)>len(set(normalized)))
    v['numeric_options']=int(bool(options) and all(re.fullmatch(r'-?\d+(?:\.\d+)?',o) for o in options))
    v['chemical_transformation']=int(bool(re.search(r'reaction|reagent|synthesi|molecules? is formed|compound.{0,40}heated',lower)))
    v['biomedical']=int(bool(re.search(r'gene|bacteri|antibiotic|protein|cellular|physiolog|drug|microbiolog|rna\b|dna\b',lower)))
    v['physics_quantities']=int(bool(re.search(r'energy|momentum|velocity|quantum|magnetic|electric|oscillat|relativis|temperature',lower)) and bool(re.search(r'\d',stem)))
    v['geometric_svg']=int('<path ' in stem)
    if v['geometric_svg']:
        match=re.search(r'd="([^"]+)"',stem)
        if match:
            coords=re.findall(r'[ML]\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)',match[1]);v['svg_many_sides']=int(len(set(coords))>=7)
    v['table_query']=int('table where the first line is a header' in lower)
    v['table_modified']=int(v['table_query'] and bool(re.search(r'we then (?:delete|add|insert)',lower)))
    v['table_filter']=int(v['table_query'] and bool(re.search(r'how many.{0,80}\band\b',lower)))
    if v['narrative_belief']:
        match=re.search(r'place (\w+) would look to find the (.+?) given the story',lower)
        if match:
            person,target=match[1],match[2]
            movement=r'(?:mov(?:e[ds]?|ing)|relocat\w*|plac(?:e[ds]?|ing)|put(?:s|ting)?|transfer\w*|transport\w*|slid(?:es?)?|carr(?:y|ies|ied|ying)|return\w*)'
            pattern=r'\b'+re.escape(person)+r'(?:,\s*[^,]{0,60},)?\s+(?:\w+ly\s+){0,2}'+movement+r'\b[^.!?]{0,100}\b'+re.escape(target)+r'\b'
            v['target_query_move_clause']=int(bool(re.search(pattern,lower)))
            for sentence in re.split(r'[.!?]\s*',lower):
                if target in sentence and re.search(r'\b(?:back|return\w*)\b',sentence):v['target_restoration_cue']=1;break
    if v['narrative_allocate']:
        v['allocation_solitary_cue']=int(bool(re.search(r'work(?:s|ing)? (?:alone|individually)|work by (?:him|her|them)self|prefers? solitude',lower)))
    return [v[key] for key,_,_ in FEATURES],diagnostics
