"""Build a detailed Word explanation of the thesis."""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()
for s in doc.sections:
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.3)
    s.right_margin = Cm(2.3)

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(8)
style.paragraph_format.line_spacing = 1.15

def shade(cell, hex_color):
    tc = cell._tePr if hasattr(cell, "_tePr") else cell._tc
    tcPr = tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:fill"), hex_color)
    sh.set(qn("w:val"), "clear")
    tcPr.append(sh)

def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        run.font.name = "Times New Roman"
    return p

def para(text, bold=False, italic=False):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    return p

def bullets(items):
    for it in items:
        p = doc.add_paragraph(it, style="List Bullet")
        for r in p.runs:
            r.font.name = "Times New Roman"
            r.font.size = Pt(12)

def add_table(headers, rows):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(head)
        r.bold = True
        r.font.size = Pt(10)
        r.font.name = "Times New Roman"
        shade(cell, "1F3A5F")
        r.font.color.rgb = RGBColor(255, 255, 255)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = t.rows[ri + 1].cells[ci]
            cell.text = ""
            r = cell.paragraphs[0].add_run(val)
            r.font.size = Pt(10)
            r.font.name = "Times New Roman"
            if ri % 2 == 1:
                shade(cell, "F4F7FB")
    doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Explanation of the Thesis")
r.bold = True
r.font.size = Pt(22)
r.font.name = "Times New Roman"
r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run(
    "Fault diagnosis of rotating machinery by vibration–wavelet–coefficient\n"
    "features and a modified shuffled frog-leaping support vector machine"
)
r.italic = True
r.font.size = Pt(13)
r.font.name = "Times New Roman"

sub2 = doc.add_paragraph()
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub2.add_run(
    "A reading guide to the expanded technical paper developed from\n"
    "You, Fan, Li, Liang, Fang, and Wang, Shock and Vibration, 2019,\n"
    "Article ID 1908485."
)
r.font.size = Pt(11)
r.font.name = "Times New Roman"

h("1. What the thesis is")
para(
    "The thesis is an expanded technical study of one published experiment. "
    "The experiment diagnoses a vertical axial-flow pump from casing vibration. "
    "The pump is observed in four states: normal running, eccentric-axle fault (EAF), "
    "bearing-pedestal fault (BPF), and sealing-ring wear fault (SRWF). "
    "Diagnosis is split into two jobs. The first job turns a long acceleration record "
    "into a short vector of 22 numbers. The second job turns that vector into one of the four labels."
)
para(
    "The first job is the feature map called VWC: vibration severity, a dyadic wavelet energy "
    "time-spectrum, and the coefficient power spectrum of the loudest wavelet level. "
    "The second job is a radial-basis support vector machine whose two parameters are chosen "
    "by a modified shuffled frog-leaping algorithm (MSFLA). "
    "These two jobs are the two innovations. Each has its own five-page paper, in the folders paper1 and paper2."
)
para(
    "The long paper is 86 pages. Chapter 1 (pages 10–29) is the research of previous work. "
    "Chapter 2 (pages 30–59) specifies the two technologies. "
    "Chapter 3 (pages 60–79) specifies the rig and reads the published classification counts. "
    "The bibliography has 77 references. "
    "This document explains that argument in order, without requiring the reader to reconstruct it from the equations."
)

h("2. Where the numbers come from")
para(
    "Two kinds of material sit side by side in the thesis, and they must not be confused."
)
para(
    "Published measurements. Vibration severities, pump duty, sensor and acquisition settings, "
    "sample counts, and classification rates are the values reported by You and co-authors in 2019. "
    "The thesis does not repeat the laboratory. It reads that experiment and writes the missing mathematics "
    "in a form that can be implemented.",
    bold=False,
)
para(
    "Illustrative figures. The time-domain traces, baseband spectra, and dyadic energy bars are synthetic signals. "
    "They were scaled so that the root-mean-square velocity matches the published severities, and they were shaped "
    "so that each state emphasizes the harmonics the 2019 paper describes. "
    "They are pictures of the idea. They are not the laboratory recordings, and no accuracy number in Chapter 3 is computed from them."
)
para(
    "One cell in the 2019 training-count table is inconsistent. For the backpropagation network on sealing-ring wear, "
    "the printed test count does not match the printed percentage, and it does not match the printed average of 85 percent. "
    "The percentage 86.667 percent is 13 out of 15, and the four BPNN percentages average to exactly 85 percent only if that cell is 13 out of 15. "
    "The thesis uses 13 out of 15. Every other cell already agrees with its percentage and with the pooled accuracy."
)

h("3. The machine and the four faults")
para(
    "The machine is a vertical axial-flow pump in a closed water loop. "
    "The stated duty is 960 revolutions per minute, a flow of 550 cubic metres per hour, a head of 3 metres, and a four-blade impeller. "
    "Shaft frequency is therefore f0 = 960/60 = 16 Hz exactly. "
    "Blade-pass frequency is 4 times 16 Hz, which is 64 Hz, the same tone as the fourth shaft harmonic. "
    "A healthy spectrum is allowed to contain energy at 64 Hz. That line is not, by itself, a fault."
)
para(
    "A capture used to describe the records is 8192 samples at 10 kHz. "
    "The record lasts 0.8192 seconds, contains about 13.1 shaft revolutions, and has a Fourier bin width of 1.221 Hz. "
    "That bin width puts 48 Hz (the third harmonic) and 50 Hz (the electrical supply) in different bins. "
    "The sensor is a triaxial charge accelerometer, B&K 4321, on the casing. "
    "The three directions are radial, tangential, and axial. The acquisition card samples at 10 kHz."
)
para("The four states are produced physically, not by simulation.")
bullets([
    "Normal: the pump at the stated duty with no intentional defect. Radial severity 1.051 mm/s. Tangential energy peaks at 16 Hz.",
    "Eccentric axle (EAF): a weight is moved on the shaft so the mass center leaves the axis. This is unbalance. Severity rises to 3.737 mm/s, about 3.56 times the healthy value. The tangential peak stays at 16 Hz and grows.",
    "Bearing pedestal (BPF): a rolling-element bearing defect loosens the pedestal that supports it. The stiffness is no longer constant inside one revolution, so a half-order line at 8 Hz appears. Severity only rises to 1.161 mm/s, about 10 percent above healthy. This fault is almost invisible if severity is the only feature.",
    "Sealing-ring wear (SRWF): the ring that limits backflow and keeps the impeller off the casing is worn. Severity is 1.700 mm/s. The spectrum emphasizes 16 Hz, 8 Hz, and 32 Hz.",
])
para(
    "The tangential channel carries the largest spectral amplitudes in every state. "
    "The axial channel is still kept, because under the pedestal fault the high orders (80 Hz and 96 Hz) grow there. "
    "The radial channel is the one for which severity is published, so the single severity coordinate is the radial severity."
)

h("4. Chapter 1 — previous work, and the problem")
para(
    "Chapter 1 has three sections. The first teaches the principles. The second reviews the technical lines that already exist. "
    "The third states the gap those lines leave on this pump."
)
h("4.1 Principles", 2)
para(
    "A linear rotor model is enough to name the forces. Unbalance produces a rotating force at shaft frequency whose size grows with the square of speed. "
    "A sharp bearing defect produces impacts. Those impacts ring the housing, so the raw spectrum shows a resonance with sidebands, not a single new tone at 16 Hz. "
    "A loose pedestal modulates stiffness and creates fractional harmonics such as half order. "
    "Seal wear changes the hydraulic load and writes energy at shaft order and at twice shaft order. "
    "A fifth force is interference: neighboring machines and the 50 Hz supply. The feature sampler has to be narrow enough not to call 50 Hz a mechanical harmonic."
)
para(
    "The instrument measures casing acceleration, not shaft displacement. Acceleration amplifies high-frequency impacts and can hide a slow unbalance until the signal is integrated back toward velocity. "
    "That is why severity is defined on velocity. Standards (ISO 10816, now ISO 20816) zone machines by root-mean-square velocity because that number tracks damage better than a raw acceleration peak."
)
para(
    "Time-domain scalars such as kurtosis and crest factor rise when rare impacts interrupt a smooth carrier, and they also rise when a single interference spike arrives. "
    "The thesis keeps only severity from the time domain. Frequency analysis by the Fourier transform is the right tool for steady sinusoids and the wrong final tool for a short impact: the transform smears the impact across the record and reports only the average. "
    "Wavelets refuse a single window. A wide scale sees the shaft harmonic. A fine scale sees the impact. "
    "Empirical mode decomposition is the adaptive alternative. The thesis explains it and then does not use it, because on only sixty records per class an adaptive mode order is an extra source of disagreement between training and test."
)
para(
    "Once the record is a vector, diagnosis is classification. The training set has 180 vectors in 22 dimensions. "
    "A richly parameterized neural network can fit those 180 rows and still fail on the 60 held-out rows. "
    "A support vector machine limits that risk by maximizing the margin. Its practical price is two continuous parameters, and those parameters are part of the method."
)
h("4.2 Previous work", 2)
para(
    "The review is organized by idea, not as a flat list. Symptom catalogues (Sohre) still supply the vocabulary: unbalance, looseness, hydraulic leakage. "
    "Model-based residuals (Beard, Willsky, Frank, sliding-mode observers, geometric isolation) are the right tools when an accurate plant model exists. "
    "This pump does not offer that model. Casing acceleration is almost the only measurement, so the study is data-driven."
)
para(
    "Wavelet energies, wavelet-packet energies, spectral kurtosis, and empirical-mode envelope spectra are the standard repairs for early impulsive faults. "
    "Neural networks trained on time-domain or spectral features work, and they drop more accuracy from training to test when the sample is small. "
    "Support vector machines are the small-sample default in this literature. "
    "Their penalty and kernel width are set by grid search, genetic algorithms, particle swarms, artificial bee colonies, chemical-reaction search, or shuffled frog-leaping. "
    "The frog-leaping algorithm is attractive because each step is simple and because it already mixes a local leader with a global leader. "
    "It is also the algorithm whose two defects the second innovation repairs."
)
h("4.3 The problem the thesis takes on", 2)
para("Four gaps remain after that literature, and they are the problem statement.")
bullets([
    "Feature gap. Severity cannot see the pedestal fault. A Fourier spectrum averages transients. A wavelet energy vector cannot tell 8 Hz from 16 Hz. The missing object is one short vector that contains intensity, scale, and shaft order, on three axes.",
    "Stationarity gap. Early faults are a strong periodic carrier plus a weak nonstationary disturbance plus interference. The front end has to see time and scale, and it still has to return shaft-order amplitudes.",
    "Search gap. Frog-leaping uses the local leader and the global leader only in sequence, and it replaces a failed frog at random. The search collapses and the random restart is blind.",
    "Evaluation gap. A pooled percentage hides the pedestal class, hides the train-to-test drop, and hides the fact that the entire test set is 60 vectors. One vector is 1.67 percentage points.",
])

h("5. Innovation 1 — the VWC feature map")
para(
    "Paper 1 is the short paper on this innovation. The long treatment is Chapter 2, Section 2.1. "
    "VWC sends three acceleration channels to one point in a 22-dimensional space. The letters stand for the three blocks that are concatenated."
)
h("5.1 Block A — vibration severity", 2)
para(
    "Severity is the root-mean-square of a velocity series. In continuous time it is the square root of the average of velocity squared. "
    "On uniform samples the integral collapses to the ordinary RMS. "
    "The 2019 formula builds each velocity sample by integrating absolute acceleration over 10 milliseconds (100 samples at 10 kHz). "
    "That is a local intensity, not a signed integral of acceleration. "
    "The thesis keeps the published severities under that definition and does not translate them into an ISO zone letter, because a different integrator would not produce the same millimetre-per-second number."
)
para(
    "Ratios to the healthy severity: eccentric axle 3.56, pedestal 1.10, seal wear 1.62. "
    "A threshold placed anywhere between 2 and 3 mm/s catches only the eccentric axle. "
    "Severity is necessary because an operator needs the intensity, and it is not sufficient because two of the three faults hide inside it."
)
h("5.2 Block B — which scale is loud", 2)
para(
    "A wavelet compares the record with stretched and shifted copies of one short waveform, the mother wavelet. "
    "Dyadic scales are powers of two. Mallat’s filter bank computes those comparisons with a low-pass filter and a high-pass filter, throwing away every second sample after each filter. "
    "The high-pass output at level j is the detail. Its energy time-spectrum is the sequence of squared detail coefficients. "
    "That sequence is high where, inside the record, that scale was loud. "
    "The scalar used to rank scales is the sum of the sequence. The winning level is the maximum. Ties go to the finer level."
)
para(
    "The exposition uses the Daubechies wavelet with four vanishing moments (db4) and eight levels. The 2019 paper does not name the mother wavelet. "
    "db4 is a stated choice of the exposition, not a recovered setting of the experiment. "
    "Changing to a longer Daubechies wavelet moves a little energy between neighboring levels. It does not change the definition of the map."
)
para(
    "Level numbering must be read against the sampling rate. At 10 kHz, detail d5 occupies roughly 156 to 312 Hz. "
    "The shaft tones from 8 to 96 Hz are coarser than that. The 2019 plots nevertheless show d5 as the loudest detail. "
    "That is physically plausible: casing acceleration amplifies a structural or fluid resonance in the few-hundred-hertz band, and the shaft orders ride on that resonance as modulation. "
    "The map recomputes the winning level for every record and every axis. It does not hard-code the number 5."
)
h("5.3 Block C — shaft orders on the loudest scale", 2)
para(
    "The third block takes the Fourier power spectrum of the energy time series at the winning level, and it samples that spectrum at the seven orders "
    "8, 16, 32, 48, 64, 80, and 96 Hz. "
    "Coefficient frequency is converted back to hertz by multiplying by the coefficient rate, which is the original sampling rate divided by 2 to the power of the level. "
    "At level 5 the coefficient rate is 312.5 Hz, the coefficient record has 256 samples, and the coefficient Nyquist frequency is 156.25 Hz. "
    "Every one of the seven orders lies below that Nyquist frequency, so all seven samples are legal. "
    "If a future record selected a much coarser winning level, the highest orders would sit above Nyquist. The specification refuses those orders instead of reporting a folded frequency as if it were real."
)
para(
    "The acceptance window around each order is half a shaft order, 4 Hz at this speed. "
    "The supply line at 50 Hz is 2 Hz above the third harmonic. The window aimed at 48 Hz does not include 50 Hz. "
    "Crosstalk tones at 28, 34, and 38 Hz, which the 2019 paper reports under the eccentric fault, are also outside the windows. "
    "A short vector that samples only the shaft family is partly protected from those tones. A long spectrum used as a raw feature would have to learn to ignore them from only 45 training rows per class."
)
h("5.4 How the 22 numbers are assembled", 2)
para(
    "The vector is: radial severity; seven tangential order amplitudes; seven axial order amplitudes; seven radial order amplitudes. "
    "The order of the blocks is a convention. Training and testing must use the same convention, because the classifier compares coordinates by index."
)
para(
    "Units do not match. Severity is in millimetres per second, near 1. A coefficient power sample is near one-thousandth. "
    "Left unscaled, distances inside the radial kernel would be severity distances, and the 21 spectral coordinates would be ignored. "
    "Every coordinate is therefore mapped to the unit interval. The minimum and maximum are computed on the training rows only and then applied to the test rows. "
    "Refitting the range on the test set would leak the test distribution into the feature and would inflate the accuracy. "
    "The 2019 paper says that normalization was used and does not say which map. The unit-interval map is the specification adopted in the thesis."
)
para("Published spectral peaks that make the pattern quantitative:")
bullets([
    "Healthy tangential peak at 16 Hz: 1.683×10^−3.",
    "Eccentric tangential peak: 3.322×10^−3.",
    "Pedestal tangential peak, now at 8 Hz: 4.701×10^−3, about 2.8 times the healthy tangential peak, while severity has only risen by 10 percent.",
    "Healthy radial peak: 4.574×10^−4. Eccentric radial peak: 2.492×10^−3.",
])
para(
    "Those four sentences are what the feature map is supposed to make visible. "
    "A correct implementation is one in which a scatter of the 22-vectors reproduces this pattern. "
    "The map itself contains no optimization and no random seed."
)

h("6. Innovation 2 — MSFLA-SVM")
para(
    "Paper 2 is the short paper on this innovation. The long treatment is Chapter 2, Section 2.2. "
    "The support vector machine is a known classifier. The innovation is the search that sets its two parameters, and the particular way that search is modified."
)
h("6.1 What the SVM is doing", 2)
para(
    "A hard-margin SVM draws the thickest empty slab that separates two classes. Only the points on the faces of the slab, the support vectors, determine the boundary. "
    "Moving a correctly classified point that sits well outside the slab does not change the answer. That is why a small sample can still determine a stable rule."
)
para(
    "Pump classes overlap, so an empty slab may not exist. The soft-margin version allows each training point a slack, and it charges a penalty C for the sum of the slacks. "
    "A small C permits a wide slab and many mistakes (under-fit). A large C forces the slacks toward zero and can memorize the training rows (over-fit)."
)
para(
    "The four states are not linearly separable in 22 dimensions after scaling. "
    "The radial kernel maps the vectors implicitly into a space where a linear slab is fitted, without ever writing that space down. "
    "The kernel parameter g controls the width. Large g makes each training point influence only its immediate neighborhood. Small g makes the decision almost linear. "
    "The search stores g, with g = 1/(2 σ^2), rather than the width σ."
)
para(
    "Four labels are handled by six binary machines, one for each pair of states (one-versus-one). "
    "A test vector is labeled by majority vote. One shared pair (C, g) is used for all six machines, and the fitness of that pair is the accuracy of the voted label, not the accuracy of a single binary machine. "
    "Searching six independent pairs would multiply the freedom of the search by six and, on 180 rows, would over-fit the folds. "
    "The 2019 paper does not name the multiclass reduction. One-versus-one is the specification in the thesis, chosen because each binary problem then has 45 rows of each of two states."
)
h("6.2 How a frog searches", 2)
para(
    "The search moves in logarithmic coordinates, so a step of fixed length means the same factor at C = 1 and at C = 1000. "
    "The box used in the exposition is C from 0.1 to 1000 and g across four decades around the scale of the unit-cube features. "
    "That box is a specification. It is not a number recovered from the 2019 run, which does not print the box."
)
para(
    "A frog is one candidate pair (C, g). Its fitness is five-fold accuracy on the training set. "
    "The folds are stratified: each fold of 36 rows contains nine rows of each state, so a frog cannot score perfectly by ignoring the pedestal class. "
    "The test set is never seen inside the fitness. After the best frog is returned, the six machines are retrained on all 180 rows and applied once to the 60 test rows."
)
para(
    "Unmodified shuffled frog-leaping deals the population into memeplexes. "
    "In each memeplex the worst frog leaps toward the best frog of that memeplex. "
    "If the leap does not improve fitness, it leaps toward the best frog of the whole population. "
    "If that also fails, it is replaced by a uniform random frog. "
    "Memplexes are then shuffled and the process repeats."
)
h("6.3 What the modification changes", 2)
para(
    "Two changes, and nothing else. The SVM, the kernel family, and the fitness stay the same. "
    "Only the proposal of the next (C, g) changes. That is why the comparison with unmodified frog-leaping is the clean estimate of the innovation."
)
para(
    "First, the worst frog learns from both leaders in one leap. "
    "The weight on the global leader is the frog’s distance to that leader, scaled between the nearest and the farthest frog. "
    "The weight on the local leader is one minus that number. "
    "A frog already near the global best mostly polishes its position using the local best. "
    "A frog far away is pulled toward the global best immediately, instead of wasting a training on a local leap that will fail. "
    "Distance includes the distance between memeplex centers, so two groups that have drifted apart are not treated as neighbors just because two of their members happen to be close."
)
para(
    "Second, a failed leap is not a random restart. The current frog is multiplied, coordinate by coordinate, by a Gaussian perturbation of mean zero and variance one, and the result is clipped back into the box. "
    "On logarithmic coordinates this is a multiplicative jitter of C and of g. "
    "A typical draw changes C by a small factor, enough to leave a flat spot of the accuracy surface and not enough to teleport across the box. "
    "The mutant is kept even if its fitness drops. That acceptance of a worse neighbor is what puts diversity back after a failure."
)
h("6.4 What is deliberately not invented", 2)
para(
    "The 2019 paper does not print the population size, the number of memeplexes, the number of local steps, the number of shuffles, the step limit, the numerical pair (C, g) that won, or the random seed. "
    "The thesis describes a legal software starting point — five memeplexes of ten frogs, ten local steps, thirty shuffles — and labels it as a suggestion for an implementation, not as the setting of the published run. "
    "The accuracy tables belong to the search the authors actually ran, under the settings they did not print."
)

h("7. Chapter 3 — the build and the estimate")
h("7.1 The build", 2)
para(
    "The processing path is the same for every classifier. Records go through VWC, then through the training-only scale. "
    "Only after that point do the four methods branch: a backpropagation network, a chemical-reaction SVM (ACROA-SVM), unmodified frog-leaping SVM (SFLA-SVM), and the modified search (MSFLA-SVM). "
    "Because the feature is shared, the table is a comparison of decision rules, not a comparison of sensors."
)
para(
    "The labeled sample is 60 groups in each of the four states, 240 groups in all. "
    "Forty-five groups per state are training and fifteen are test. "
    "The matrices are 180 by 22 and 60 by 22. "
    "The 2019 text does not say whether the fifteen test groups were drawn at random. A repetition should draw them at random inside each state and freeze the draw before any search over C and g."
)
para(
    "One test vector moves a per-class rate by 6.67 points and the pooled rate by 1.67 points. "
    "Any discussion of a one-point pooled gap that does not mention this denominator is overstating the experiment."
)
para(
    "The sensor responds up to 12 kHz and the converter samples at 10 kHz, so energy between 5 and 12 kHz can fold into the analysis band unless an analog anti-alias filter stops it. "
    "The 2019 table does not list that filter. The thesis treats the filter as a required part of any repetition and as an open term in the error budget."
)
h("7.2 Training results", 2)
para("Correct training labels out of 45 per state, then the pooled count out of 180.")
add_table(
    ["State", "BPNN", "ACROA-SVM", "SFLA-SVM", "MSFLA-SVM"],
    [
        ["Normal", "42/45", "45/45", "45/45", "45/45"],
        ["EAF", "41/45", "44/45", "43/45", "45/45"],
        ["BPF", "40/45", "42/45", "42/45", "43/45"],
        ["SRWF", "42/45", "44/45", "44/45", "44/45"],
        ["Pooled", "165/180 (91.7%)", "175/180 (97.2%)", "174/180 (96.7%)", "177/180 (98.3%)"],
    ],
)
para(
    "The healthy state is the easiest training class. Every SVM variant classifies all 45 healthy training rows correctly. "
    "The pedestal state is the hardest training class for every method. "
    "The modified search leads or ties on every state. Its pooled lead over unmodified frog-leaping is three training vectors."
)
h("7.3 Test results", 2)
para("Correct test labels out of 15 per state, then the pooled count out of 60. This is the number the search was not allowed to see.")
add_table(
    ["State", "BPNN", "ACROA-SVM", "SFLA-SVM", "MSFLA-SVM"],
    [
        ["Normal", "13/15", "15/15", "15/15", "15/15"],
        ["EAF", "13/15", "13/15", "14/15", "14/15"],
        ["BPF", "12/15", "13/15", "13/15", "13/15"],
        ["SRWF", "13/15", "14/15", "13/15", "14/15"],
        ["Pooled", "51/60 (85.0%)", "55/60 (91.7%)", "55/60 (91.7%)", "56/60 (93.3%)"],
    ],
)
para(
    "In words: the modified search makes 4 test errors, each of the other two SVM searches makes 5, and the neural network makes 9. "
    "The entire pooled gain of the modification over unmodified frog-leaping is one sealing-ring vector. "
    "On the pedestal class every SVM method misses 2 out of 15. The modification does not create a new success there. "
    "It does collect, in one rule, the eccentric-axle count of unmodified frog-leaping (one error) and the seal-wear count of the chemical-reaction search (one error)."
)
para(
    "The drop from training accuracy to test accuracy is about 6.7 points for the neural network and 5.0 points for both frog-leaping machines. "
    "The modified search has the highest training accuracy and the highest test accuracy, and its drop is not larger than the unmodified leap’s drop. "
    "That is the pattern expected if the search has not bought its training score by memorizing the folds. "
    "On a sample of this size the equality of the two drops is one or two vectors, not a theorem."
)
h("7.4 How large the differences are", 2)
para(
    "A reader who assumes independent coin-flips with success probability 0.933 and 60 trials gets a standard deviation of about 3 percentage points on the pooled accuracy. "
    "Vectors from the same physical run are not independent, so this is only a scale. "
    "It is enough to see that the one-vector gap between the SVM searches sits inside ordinary variation, while the five-vector gap against the neural network sits at the edge of it. "
    "The thesis reports both tables and does not promote the one-vector gap into a slogan detached from the denominator. "
    "No confidence interval is invented."
)
para(
    "The practical recommendation in the thesis is correspondingly modest. "
    "Use the 22-vector rather than severity alone, because the pedestal fault is invisible to severity. "
    "Use a radial SVM rather than the unconstrained neural baseline on 180 training vectors, because the test count is higher and the drop is smaller. "
    "Use the modified leap rather than the unmodified leap if the implementation cost is the same, and do not expect more than a vector or two on a test set of sixty. "
    "Spend the next experimental effort on interference in the pedestal records, which is where all four methods already agree that the vector is weakest."
)

h("8. What the thesis does not claim")
bullets([
    "It does not claim a new laboratory campaign. Severities, hardware, and classification counts come from the 2019 paper.",
    "It does not claim that the waveform figures are measured signals.",
    "It does not claim an optimal numerical pair (C, g), because that pair was not published.",
    "It does not claim that db4 was the wavelet of the 2019 run. db4 is the wavelet of the specification and of the illustrative energy figure.",
    "It does not claim performance at any speed other than 960 r/min, at any other flow, under mixed faults, or as a remaining-life prediction.",
    "It does not claim that the modified leap beats particle swarm, genetic algorithms, or bee-colony search on this pump. Those searches are not in the published tables.",
    "It does not claim which of the 22 coordinates is necessary. No ablation was published.",
])

h("9. How the files are organized")
add_table(
    ["File", "Role"],
    [
        ["1.pdf", "The 2019 source article (17 pages)."],
        ["1_extracted.txt", "Plain text extracted from 1.pdf."],
        ["paper/main.tex", "LaTeX source of the 86-page thesis. Chapters are ch1.tex, ch2.tex, ch3.tex, plus the later sections ch2extra.tex and ch3extra.tex."],
        ["Fault_Diagnosis_Paper.pdf", "The compiled thesis."],
        ["paper.txt", "Plain text extracted from the thesis PDF."],
        ["paper1/", "Five-page paper on innovation 1, the VWC feature."],
        ["paper2/", "Five-page paper on innovation 2, the MSFLA-SVM search."],
        ["Thesis_Explanation.docx", "This document."],
    ],
)
para(
    "To recompile the thesis, run pdflatex twice inside the paper folder. "
    "To recompile either short paper, run pdflatex twice inside paper1 or paper2. "
    "The short papers read two figures from paper/figs, so that folder has to stay in place."
)

h("10. A short way to remember the argument")
para(
    "The pump has one intensity fault and two shape faults. Eccentricity is the intensity fault: the machine simply vibrates about three and a half times as hard, at shaft frequency. "
    "The pedestal fault and the seal-wear fault are shape faults: the energy moves among shaft orders, especially toward half order, while the severity barely changes or changes only moderately. "
    "VWC is the feature that can see both kinds, because it stores the intensity and the orders. "
    "A radial SVM is the classifier that can draw the boundary from 180 examples without the large train-to-test drop of the neural baseline. "
    "The modified frog leap is a careful way to pick the SVM’s two knobs, using both the local and the global leader on every try, and mutating locally when a try fails. "
    "On the published test set that whole chain makes four mistakes out of sixty. Two of the four are pedestal cases. Those two are the part of the problem that is still open."
)

out = r"C:\Users\Administrator\Workspace\Own\Thesis\vs2\kgh\Thesis_Explanation.docx"
doc.save(out)
print("saved", out)
