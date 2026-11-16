# Explanation of the thesis

This note explains the thesis in `main/thesis.pdf`. It is a reading guide, not a second thesis. The thesis itself is a computer-science manuscript on system design and optimization. The plant it designs is the wastewater end of a brewery: water that was used in production is cleaned and then discharged. The machines that dominate the electricity bill, and that the two new methods control or inform, are fans.

## What problem the thesis is solving

A brewery does not produce a smooth municipal sewage flow. It produces batches. For long stretches the waste load is moderate. Then a brewhouse peak arrives. Then a clean-in-place cycle dumps hot water that contains detergents. Those detergents are surfactants. They sit on air bubbles and reduce the rate at which oxygen moves from the air into the water. The aerobic polishing tank still has to meet a discharge limit on chemical oxygen demand. If dissolved oxygen collapses, treatment slows. If the fans run as if every hour were the worst hour, most of the electricity is wasted.

The thesis treats this as an automation and optimization problem, not as a new biological process. The treatment train is conventional and is stated as a representative design, not as the nameplate of a real factory:

1. Screening.
2. Equalization, with exhaust fans so the tank does not go septic and odorous.
3. pH correction.
4. Anaerobic pretreatment, which is how a brewery usually cuts a very strong raw waste down to a polishing load. This stage is included so the aerobic fans are not asked to oxidize raw beer waste.
5. Dissolved-air flotation as a solids buffer, again with an exhaust fan.
6. An aerobic polishing tank of 900 cubic metres, with fine-bubble diffusers and three variable-speed centrifugal blowers on one header.
7. A clarifier, then discharge.

The study discharge target used inside the optimization is 125 milligrams per litre of chemical oxygen demand. That number is a study permit, chosen at the same order as a common secondary-treatment figure. A real site permit can differ. The controller needs some numeric limit. It does not depend on the limit being exactly 125.

The week used for the estimate has five regimes that sum to 168 hours: 96 hours of base production, 30 hours of brewhouse peak, 18 hours of fermentation clean-in-place (where the oxygen-transfer factor α falls from about 0.60 to 0.40), 12 hours of packaging clean-in-place, and 12 hours of weekend hold. Mean aerobic influent flow is about 55.5 cubic metres per hour. The flow-weighted mean chemical oxygen demand into the aerobic tank is about 402 milligrams per litre. Raw brewery waste is often several thousand milligrams per litre. The thesis starts after anaerobic pretreatment has already done the bulk removal.

## The two innovations

The thesis has exactly two technological claims. Everything else is the plant model, the automation architecture, and the estimate that those two claims are worth making.

### Innovation 1 — schedule-aware fan control

Municipal aeration controllers, including published model-predictive and economic model-predictive controllers, usually see a daily sewage pattern and a smooth airflow command. A brewery already knows, from its production program, when a clean-in-place recipe will start. Innovation 1 uses that calendar as a measured future disturbance.

The controller sits on an edge computer and publishes, once a minute, how many blowers are on and what header airflow they should deliver. The programmable controller keeps the safety interlocks and the fast inner loops. If the edge computer stops, the plant falls back to a dissolved-oxygen proportional-integral loop. That split is deliberate. An optimizer that is the only device able to start a fan is not an operable plant.

The deployed law, which is what the numerical week actually integrates, does four things each minute:

- It reads the regime now and the regime 0.75 hours ahead. The lead covers the blower lag (about 0.1 hour) and the dissolved-oxygen probe lag (about 0.08 hour).
- It raises the dissolved-oxygen set-point before a harder regime. The set-points are 1.8 milligrams per litre in base production, 2.0 in the brewhouse peak and packaging clean-in-place, 2.2 in fermentation clean-in-place, and 1.3 on the weekend. A permanent operator set-point of 2.5 is the baseline, not the proposed law. Holding 2.5 all week buys a smaller driving force and therefore more air for the same oxygen uptake.
- It scales the oxygen-uptake estimate by the ratio of previewed mass load to current mass load, clipped so the preview cannot more than modestly inflate the command.
- It inverts the oxygen-transfer curve to get the airflow that meets that uptake at the set-point, using the current oxygen-transfer factor α, then trims the command with a clamped integral bias so a constant probe offset does not leave a constant dissolved-oxygen error.

Staging is a rule while the three machines are identical: one machine below 2600 normal cubic metres per hour, two machines up to 5200, three above that. Each online machine must stay above a surge margin. The mixing floor is 900 normal cubic metres per hour, which is high enough that one machine is legal and two machines sharing that flow would not be. The weekend therefore runs one blower. The energy left on the table there is the price of not crossing the surge line.

The thesis also writes the fuller problem the deployed law is a reduction of: a hybrid economic model-predictive controller over a horizon, with power, dissolved-oxygen hinge penalties, discharge hinge penalties, and a cost on starting or stopping machines. When the machines are no longer identical, that integer problem, or three convex programs enumerated over the machine count, replaces the staging rule. The week of numbers was not produced by that solver. The thesis says so. The numbers come from the one-step inversion above.

Exhaust fans are in the automation and in the energy account. They are not decision variables of the blower optimizer. Their speed follows the same calendar by a cube law, because they discharge into a duct. Two fans rated 7.5 kilowatts each use about 1300 kilowatt-hours in the representative week. That is about 12 percent of the proposed plant’s fan electricity. The savings the thesis quotes are blower savings. Mixing the exhaust load into the percentage would make the blower result look smaller for no physical reason.

### Innovation 2 — physics-informed oxygen-transfer sensing

A dissolved-oxygen probe reports concentration. It does not report why the concentration moved. A surfactant event lowers the transfer coefficient. A load event raises oxygen uptake. Both make the probe fall, and a proportional-integral fan loop adds air for both. If the probe has fouled and reads low, the same loop adds air that the tank does not need.

Off-gas hoods can measure transfer directly. They are the right commissioning instrument. They are a weak assumption as the only signal on every detergent day. Innovation 2 is a soft sensor that runs on instruments the fans already have: airflow, electrical power, header pressure, temperature, influent flow, the probe, and the detergent flag from the calendar.

The network is small. Two hidden layers of 32 units emit four heads: the transfer factor α, the transfer coefficient, the oxygen uptake, and a one-step dissolved-oxygen prediction used only for training. Only the first three are published to the fan controller. Training uses three residuals, not only the probe error:

- The oxygen mass balance must close.
- Measured electrical power must agree with the fan map at the claimed airflow.
- The emitted transfer coefficient must lie on the clean-water curve at the emitted α.

The third residual is the one that stops a non-uniqueness. Without it, many pairs of “low transfer” and “low uptake” fit the same probe trace. With it, α is the free transfer parameter and the balance solves for uptake. A health bit clears if the residual is large, if α leaves a plausible range, or if the uptake head moves faster than the biology can justify. When the bit is clear, the controller uses the regime table (0.40 in the fermentation recipe, not the quiet-week 0.55) instead of the network.

In the numerical week the sensor is not a particular trained network with a random residual. It is the true regime α, passed through a 0.30 hour filter and a two-percent caution. The thesis calls this an upper bound: it answers what the information is worth if the sensor is good. It does not answer what residual a given training run will leave. Commissioning against an off-gas or portable probe on a detergent day is still required before that bound is treated as a site forecast.

## How the two innovations depend on each other

Innovation 1 without a live α still has the calendar and the set-point table. On this week that is where most of the remaining energy saving against a habitual 2.5 milligram-per-litre loop comes from. Innovation 2 without a controller that has a place to put α is only a forecast of dissolved oxygen. The thesis defines the sensor as solved only when its α and transfer coefficient enter the fan command.

The degraded modes are specified, not left as an exercise:

- Sensor residual high: use the regime table, keep the calendar.
- Probe stuck: freeze airflow and alarm. Do not let the integral wind up against a dead measurement.
- Calendar lost: stop previewing, keep the sensor.
- Both lost: hold the last known regime table and hold the bias.

## What the estimate found

Four closed loops were integrated for 168 hours on the same reduced aerobic model. Substrate, inert chemical oxygen demand, ammonia, and dissolved oxygen are states. Mixed-liquor solids are held constant by sludge-age control. Fan power is polytropic compression against a header whose pressure is mostly the static submergence of the diffusers, plus a smaller pipe-friction term. A pure cube law is the wrong cost for these blowers, because the water depth does not grow with fan speed. The cube law is kept only for the exhaust fans.

Blower electricity over the week:

| Loop | Energy | Mean power | Hours above 125 mg/L COD |
|---|---:|---:|---:|
| Proposed (calendar + live α) | 9656 kWh | 57.5 kW | 0 |
| Predictive inversion with α locked at 0.55 and no calendar scale | 9620 kWh | 57.3 kW | 0 |
| Proportional-integral at a fixed 2.5 mg/L | 10920 kWh | 65.0 kW | 0 |
| Constant speed at 6552 Nm³/h, sized for the detergent hour with a margin | 31151 kWh | 185.4 kW | 0 |

Against constant speed, the proposed loop uses 21495 kilowatt-hours less, which is 69 percent. Against the proportional-integral fallback, it uses 1264 kilowatt-hours less, which is 11.6 percent. Against the fixed-α inversion it uses about 36 kilowatt-hours more. That last comparison is the point of innovation 2. The energy is almost the same. The dissolved-oxygen floor is not.

In the fermentation clean-in-place window the minima are:

- Proposed, with the lagged true α: 1.63 mg/L.
- Fixed α of 0.55: 0.90 mg/L.
- Proportional-integral at 2.5 mg/L: 1.87 mg/L.
- Constant speed: 2.90 mg/L.

The operational floor used in the study is 1.5 mg/L during production. The fixed-α law crosses it. The proposed law does not, in that window. The proportional-integral law also stays above the floor, and pays for that in the 96 quiet hours by holding 2.5 mg/L when 1.8 mg/L already meets the discharge limit. About half of its weekly penalty against the proposed loop is those quiet hours.

Two boundary dips of the proposed loop are reported and not smoothed away. Entering the brewhouse peak, dissolved oxygen touches 1.39 mg/L. Entering the weekend, it touches 0.86 mg/L for a short interval, while the weekend mean is 4.46 mg/L because the mixing floor then over-aerates. The thesis names the correction: allow the load scale to fall below one when the previewed load falls, and lengthen the lead on large upward steps. Those corrections were not re-integrated. The tables are the evidence that they are worth doing. They do not create hours above the chemical-oxygen-demand limit on this week.

An open-loop calculation, separate from the week, freezes airflow and freezes uptake at 30 milligrams per litre per hour after the detergent arrives. Airflow computed with α = 0.55 (about 3570 Nm³/h) lets dissolved oxygen fall from 1.80 to about 0.53 in 0.8 hours. Airflow computed with α = 0.40 (about 5250 Nm³/h) climbs toward 2.2 and stays there. The closed-loop week does not reach 0.53, because uptake falls as dissolved oxygen falls and because the bias adds air. The open-loop table is the unmitigated error. The week minimum of 0.90 is the error after that mitigation. The gap from 0.90 to 1.63 is what the live factor is for.

Specific blower energy of the proposed loop is 1.03 kilowatt-hours per cubic metre of aerobic influent, or 2.58 kilowatt-hours per kilogram of aerobic influent chemical oxygen demand. Adding the exhaust fans raises the all-fan figure to about 1.17 kilowatt-hours per cubic metre. At an illustrative 0.12 currency units per kilowatt-hour, used only as a conversion, the constant-speed gap is on the order of 130000 currency units a year if every week looked like this one. The thesis does not claim that price is a tariff.

## How the three chapters are organized

Chapter 1 is previous work and the problem statement. Section 1 reviews aeration energy, fan maps, surge, proportional-integral and ammonia-based aeration control, and model-predictive aeration control on the standard benchmark plants. Section 2 reviews off-gas measurement, the surfactant factor, soft sensors, and physics-informed neural networks. Section 3 writes the plant, the week, the oxygen and substrate balances, and the two decision problems: P1 for the fans and P2 for the sensor. The gap it states is precise. Published predictive aeration controllers do not ingest a brewery recipe clock, rarely put centrifugal surge inside the same optimization as dissolved oxygen, and treat α as a constant. Published physics-informed dissolved-oxygen models stop at a forecast of concentration. They do not hand α to a fan optimizer.

Chapter 2 specifies both technologies. The first section is the schedule encoder, the prediction model, the surge and power constraints, the economic cost, the deployed minute-by-minute law, a hand inversion at the base load (about 1550 Nm³/h and about 33 kW), and the exhaust-fan rule. The second section is the sensor: inputs, heads, loss, identifiability, the handshake with the fan law, and the degraded modes. Later sections of the chapter work the power map at six airflows, stage the machines at three loads, repeat the inversion for the detergent window, and linearize the dissolved-oxygen balance so a later quadratic program has an explicit model. That linear program is an upgrade path. It is not what produced the tables.

Chapter 3 builds the system and estimates it. The build is a four-level automation stack. Level 0 is drives and sensors. Level 1 is the programmable controller and is the only level that may energize a fan. Level 2 is the edge computer. Level 3 is the historian and the schedule import. Signals the two methods add are listed: airflow, header pressure, power per machine, speed, dissolved oxygen, temperature, the regime flag, and the schedule message. Interlocks override the optimizer: closed outlet valve, drive fault, header over-pressure, very low dissolved oxygen, and exhaust fans tied to tank level. The estimation section is the week. A later section splits every loop by regime, checks that the regime energies sum to the week totals, and states what would falsify the estimate: a header whose power does not match the map, a detergent that drives α so low that three machines cannot hold the floor, a production clock that is routinely wrong, or a permit that is actually on suspended solids rather than on chemical oxygen demand.

The implementation order follows the tables. First move constant-speed tanks onto the proportional-integral fallback. That takes the large share of the energy and does not need an edge computer. Then install the set-point table and the calendar, still with α from the regime table. Then train the sensor against a detergent-day measurement and only then close the live-α path. Building them in the reverse order would put a network in the loop before the plant has a fallback.

## What the thesis does not claim

It is not a trial on a named brewery. There is no off-gas campaign in the document. The clarifier is not dynamically modelled. Mixed-liquor solids do not change over the week. The three blowers are identical, which is why a staging rule is enough. A wrong recipe name is a different experiment from a correct calendar. Electricity price is constant, so the economic objective and the energy objective coincide apart from the constraint penalties. The percentages 11.6 and 69 belong to this parameter set and this week. A plant that already runs a low dissolved-oxygen set-point has less left to save against the proportional-integral column, and still has the constant-speed column if any tank is left in manual at the worst-hour valve.

## Files

- `main/thesis.pdf` is the thesis.
- `main/thesis.txt` is the text extracted from that PDF.
- `main/thesis.tex` and `main/chapters/` are the LaTeX sources.
- `main/references.bib` is the bibliography (103 entries in the compiled list).
- `main/estimation/simulate_week.py` is the integrator that produced the week totals. The tables in Chapter 3 match that script’s proposed, proportional-integral, fixed-α, and constant-speed runs.
- `paper1/` is a five-page paper on innovation 1 only.
- `paper2/` is a five-page paper on innovation 2 only.

Read Chapter 3 of the thesis if you want the numbers with the regime split. Read Chapter 2 if you want the equations a programmer would implement. Read this note if you want the argument in one pass before either of those.
