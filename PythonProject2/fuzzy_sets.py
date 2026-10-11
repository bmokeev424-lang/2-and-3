import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

delivery_delay  = ctrl.Antecedent(np.arange(0, 31, 0.1),   'delivery_delay')
quality_defects = ctrl.Antecedent(np.arange(0, 21, 0.1),   'quality_defects')
completeness    = ctrl.Antecedent(np.arange(70, 101, 0.1), 'completeness')
price_deviation = ctrl.Antecedent(np.arange(-15, 16, 0.1), 'price_deviation')
payment_delay   = ctrl.Antecedent(np.arange(0, 61, 0.1),   'payment_delay')

reliability = ctrl.Consequent(np.arange(0, 101, 0.1), 'reliability')

delivery_delay['low']    = fuzz.trimf(delivery_delay.universe, [0, 0, 10])
delivery_delay['medium'] = fuzz.trimf(delivery_delay.universe, [6, 15, 24])
delivery_delay['high']   = fuzz.trimf(delivery_delay.universe, [20, 30, 30])

quality_defects['low']    = fuzz.trimf(quality_defects.universe, [0, 0, 8])
quality_defects['medium'] = fuzz.trimf(quality_defects.universe, [4, 10, 16])
quality_defects['high']   = fuzz.trimf(quality_defects.universe, [12, 20, 20])

completeness['low']    = fuzz.trimf(completeness.universe, [70, 70, 85])
completeness['medium'] = fuzz.trimf(completeness.universe, [80, 88, 96])
completeness['high']   = fuzz.trimf(completeness.universe, [92, 100, 100])

price_deviation['negative'] = fuzz.trimf(price_deviation.universe, [-15, -15, 0])
price_deviation['neutral']  = fuzz.trimf(price_deviation.universe, [-6, 0, 6])
price_deviation['positive'] = fuzz.trimf(price_deviation.universe, [0, 15, 15])

payment_delay['low']    = fuzz.trimf(payment_delay.universe, [0, 0, 18])
payment_delay['medium'] = fuzz.trimf(payment_delay.universe, [12, 30, 48])
payment_delay['high']   = fuzz.trimf(payment_delay.universe, [42, 60, 60])

reliability['low']    = fuzz.trimf(reliability.universe, [0, 0, 40])
reliability['medium'] = fuzz.trimf(reliability.universe, [30, 50, 70])
reliability['high']   = fuzz.trimf(reliability.universe, [60, 100, 100])

rules = [

    ctrl.Rule(delivery_delay['low'] & quality_defects['low'] &
              completeness['high'] & payment_delay['low'], reliability['high']),
    ctrl.Rule(delivery_delay['low'] & quality_defects['low'] &
              price_deviation['neutral'], reliability['high']),
    ctrl.Rule(completeness['high'] & payment_delay['low'] &
              quality_defects['low'], reliability['high']),
    ctrl.Rule(delivery_delay['low'] & completeness['high'] &
              price_deviation['neutral'] & payment_delay['low'], reliability['high']),
    ctrl.Rule(quality_defects['low'] & completeness['high'] &
              price_deviation['neutral'], reliability['high']),

    ctrl.Rule(delivery_delay['medium'] & quality_defects['medium'], reliability['medium']),
    ctrl.Rule(completeness['medium'] & payment_delay['medium'], reliability['medium']),
    ctrl.Rule(price_deviation['neutral'] & quality_defects['medium'], reliability['medium']),
    ctrl.Rule(delivery_delay['medium'] & completeness['medium'] &
              payment_delay['low'], reliability['medium']),
    ctrl.Rule(price_deviation['negative'] & quality_defects['low'] &
              completeness['high'], reliability['medium']),

    ctrl.Rule(delivery_delay['high'] | quality_defects['high'], reliability['low']),
    ctrl.Rule(completeness['low'] | payment_delay['high'], reliability['low']),
    ctrl.Rule(price_deviation['positive'] & delivery_delay['high'], reliability['low']),
    ctrl.Rule(quality_defects['high'] & payment_delay['high'], reliability['low']),
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

def evaluate(dd, qd, comp, pdev, payd):

    sim.input['delivery_delay']  = float(np.clip(dd, 0, 30))
    sim.input['quality_defects'] = float(np.clip(qd, 0, 20))
    sim.input['completeness']    = float(np.clip(comp, 70, 100))
    sim.input['price_deviation'] = float(np.clip(pdev, -15, 15))
    sim.input['payment_delay']   = float(np.clip(payd, 0, 60))

    try:
        sim.compute()
        idx = sim.output['reliability']
    except (KeyError, ValueError):
        idx = 50.0
    mu_low    = fuzz.interp_membership(reliability.universe, reliability['low'].mf, idx)
    mu_medium = fuzz.interp_membership(reliability.universe, reliability['medium'].mf, idx)
    mu_high   = fuzz.interp_membership(reliability.universe, reliability['high'].mf, idx)

    return idx, (mu_low, mu_medium, mu_high)
def verdict(idx):

    if idx >= 60:
        return "ВЫСОКАЯ"
    elif idx >= 40:
        return "СРЕДНЯЯ"
    else:
        return "НИЗКАЯ"