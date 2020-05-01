import sys

ZERO = "Zero".upper()
Aware = "Aware".upper()
Considering = "Considering".upper()
Experiencing = "Experiencing".upper()
Ready = "Ready".upper()
Satisfied = "Satisfied".upper()
Lost = "LOST".upper()

Demo, Video, Testimonial, Pricing, Blog, Payment = "Demo Video Testimonial Pricing Blog Payment".upper().split(" ")

states = [ZERO, Aware, Considering, Experiencing, Ready, Lost, Satisfied]  # states
observes = [Demo, Video, Testimonial, Pricing, Blog, Payment]           # observations

transition_prob = {                 # transition probabilities
    ZERO: {
        Aware: 0.4,
    },
    Aware: {
        Considering: 0.3,
        Ready: 0.01,
        Lost: 0.2
    },
    Considering: {
        Experiencing: 0.2,
        Ready: 0.02,
        Lost: 0.3
    },
    Experiencing: {
        Ready: 0.3,
        Lost: 0.3
    },
    Ready: {
        Lost: 0.2,
        Satisfied: 0.4
    }

}

emission_prob = [                           # emission probabilities, row is the state seq, and col is observation seq
    [0.1, 0.01, 0.05, 0.3, 0.5, 0.0],
    [0.1, 0.01, 0.15, 0.3, 0.4, 0.0],
    [0.2,  0.3, 0.05, 0.4, 0.4, 0.0],
    [0.4,  0.6, 0.05, 0.3, 0.4, 0.0],
    [0.05, 0.75, 0.35, 0.2, 0.4, 0.0],
    [0.01, 0.01, 0.03, 0.05, 0.2, 0.0],
    [0.4, 0.4, 0.01, 0.05, 0.5, 1.0]
]

start_state = {ZERO: 1}     # all start from ZERO


def read_observes():
    obs = []
    start = False
    with open(sys.argv[1]) as f:
        for line in f.read().split("\n"):       # read data
            if line.startswith("#"):
                if not start:
                    continue
                else:
                    break               # if read # again, stop read
            else:
                start = True
                obs.append(line)
    return obs


def get_prob(state, ob):

    ps = emission_prob[states.index(state)]
    if len(ob) == 0:        # if empty line, return  (1-p1) * (1-p2) ...
        prob = 1
        for p in ps:
            prob *= (1-p)
        return prob
    else:
        prob = 1
        for o in ob:
            prob *= ps[observes.index(o)]       # return px*py...
        return prob


def B(state, ob):
    return get_prob(state, ob)


def get_start_prob(state):
    if state == ZERO:
        return 1
    else:
        return 0


def A(s1, s2):          # get the trans prob
    if s1 not in transition_prob:
        return 0
    prob = transition_prob[s1]
    if s2 == s1:
        return 1 - sum(prob.values())
    else:
        if s2 in prob:
            return prob[s2]
        return 0


if __name__ == '__main__':
    obs = read_observes()
    obs = [[z for z in x.split(",") if z != ""] for x in obs]
    T = len(obs)

    T1 = [[0 for _ in range(T)] for _ in range(len(states))]
    T2 = [[0 for _ in range(T)] for _ in range(len(states))]

    for state in states:
        si = states.index(state)
        T1[si][0] = get_start_prob(state) * B(state, obs[0])
        T2[si][0] = 0

    for i in range(1, T):
        for state in states:
            probs = []

            for k in states:
                p = T1[states.index(k)][i-1] * A(k, state) * B(state, obs[i])
                probs.append(p)

            T1[states.index(state)][i] = max(probs)
            T2[states.index(state)][i] = probs.index(max(probs))

    path = [max([i for i in range(len(states))], key=lambda x: T1[x][T-1])]
    # print(path)
    for i in range(T-1, 0, -1):
        path.append(T2[path[-1]][i])
    # print(path)
    path = path[::-1]
    for p in path:
        print(states[p])
