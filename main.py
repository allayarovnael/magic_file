import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.optimize import minimize
from functools import reduce

# initial set of media labels:
MEDIA_LABELS = ['TV', 'PZ', 'PL', 'RD', 'TZUER', 'TZR', 'KIN', 'OLDBD', 'OLMBD', 'OLV', 'SOM', 'YTB']


def adstock(data, adstock_rate):
    data_adstocked = []
    for i in range(len(data)):
        if i == 0:
            data_adstocked.append(data[i])
        else:
            data_adstocked.append(data[i] + adstock_rate * data_adstocked[i - 1])
    return data_adstocked


def hyperbel(data, hyperbel, justage=10 ** 3):
    data_hyperbel = (data / justage) / ((data / justage) + hyperbel)
    return data_hyperbel


def ots_weighting(ots_beta, ots_steigung, ots_konstante, spendings):
    base_impact_ots_one = 1 / (1 + ots_beta)
    ots = ots_konstante + ots_steigung * spendings * 1000
    impact = ots / (ots + ots_beta)
    impact_index = impact / ots if ots else 0
    ots_weight = impact_index / base_impact_ots_one
    return ots_weight


def optimized_roi_plus(scenario, media_flighting, normalize=True):
    coef = 1
    media_params = scenario.copy()
    flighting_plan = media_flighting.copy()

    media_params['OTS-weighting'] = media_params.apply(
        lambda x: ots_weighting(ots_beta=x["OTS-beta"], ots_steigung=x["OTS_Steigung"],
                                ots_konstante=x["OTS_Konstante"],
                                spendings=4*x["Budget"] / x["Anzahl_KW"]), axis=1)

    # on-top budget and ot-ots-weighting:
    media_params['OTS-weighting_OT'] = media_params.apply(
        lambda x: ots_weighting(ots_beta=x["OTS-beta"], ots_steigung=x["OTS_Steigung"],
                                ots_konstante=x["OTS_Konstante"],
                                spendings=4*(x["Budget"] + 1) / x["Anzahl_KW"]), axis=1)

    Budget_pro_Woche = dict(media_params[['Media', 'Budget_pro_Woche']].values.tolist())
    Budget_pro_Woche_OT = dict(media_params[['Media', 'Budget_pro_Woche_OT']].values.tolist())
    Effectiver_CpG = dict(media_params[['Media', 'Effectiver_CpG']].values.tolist())
    Effectiver_CpG_OT = dict(media_params[['Media', 'Effectiver_CpG_OT']].values.tolist())
    OTS_Weighting = dict(media_params[['Media', 'OTS-weighting']].values.tolist())
    OTS_Weighting_OT = dict(media_params[['Media', 'OTS-weighting_OT']].values.tolist())
    Adstocks = dict(media_params[['Media', 'Adstock']].values.tolist())
    Hyperbels = dict(media_params[['Media', 'Hyperbel']].values.tolist())
    Coefficients = dict(media_params[['Media', 'Koeffizient']].values.tolist())

    # Allocation of total media budget across weeks:
    df = flighting_plan.assign(**Budget_pro_Woche).mul(flighting_plan)  # multiply by weekly budget
    df = df.assign(**Effectiver_CpG).rdiv(df)  # multiply by effective CpG
    df = df.assign(**OTS_Weighting).mul(df)  # OTS Weighting

    # ON-TOP:
    df_ot = flighting_plan.assign(**Budget_pro_Woche_OT).mul(flighting_plan)  # multiply by weekly budget
    df_ot = df_ot.assign(**Effectiver_CpG_OT).rdiv(df_ot)  # multiply by effective CpG_OT
    df_ot = df_ot.assign(**OTS_Weighting_OT).mul(df_ot)  # OTS Weighting

    # adstock & hyperbel & coefficient multiplication:
    for col in df.columns:
        df[col] = adstock(data=df[col], adstock_rate=Adstocks[col])
        df[col] = hyperbel(data=df[col], hyperbel=Hyperbels[col])
        df[col] = df[col] * coef
        # the same for on-top:
        df_ot[col] = adstock(data=df_ot[col], adstock_rate=Adstocks[col])
        df_ot[col] = hyperbel(data=df_ot[col], hyperbel=Hyperbels[col])
        df_ot[col] = df_ot[col] * coef

    # Optimal coefficients (equal ROIplus):
    optimal_coeffs = 1 / (pd.DataFrame(df_ot.sum()) - pd.DataFrame(df.sum()))
    optimal_coeffs = optimal_coeffs.reset_index().rename(columns={"index": "Media", 0: "Koeffizient"})
    optimal_coeffs = optimal_coeffs.replace([np.inf, -np.inf], 0)
    Coefficients = dict(optimal_coeffs.values.tolist())

    # normalization:
    def div_d(my_dict):
        '''
        scale values in dict: divide each element of dict by it's max value
        '''
        sum_p = max(my_dict.values())
        if sum_p != 0:
            for i in my_dict:
                my_dict[i] = 100 * float(my_dict[i] / sum_p)
        return my_dict

    if normalize:
        Coefficients = div_d(Coefficients)  # normalize coefs:

    Coefficients = pd.DataFrame({'Media': list(Coefficients.keys()), 'Koeffizient_opt': list(Coefficients.values())})
    Wirkungsbeitraege = pd.DataFrame({'Media': df.sum().index, 'Wirkungsbeitrag': df.sum().values})
    Wirkungsbeitraege_OT = pd.DataFrame({'Media': df_ot.sum().index, 'Wirkungsbeitrag_OT': df_ot.sum().values})

    dfs = [media_params, Coefficients, Wirkungsbeitraege, Wirkungsbeitraege_OT]
    media_params = reduce(lambda left, right: pd.merge(left, right, on='Media', how='left'), dfs)
    media_params['Wirkungsbeitrag'] = media_params['Wirkungsbeitrag'] * media_params['Koeffizient_opt']
    media_params['Wirkungsbeitrag_OT'] = media_params['Wirkungsbeitrag_OT'] * media_params['Koeffizient_opt']
    media_params['ROI'] = media_params['Wirkungsbeitrag'] / media_params['Budget']
    media_params['ROI+'] = (media_params['Wirkungsbeitrag_OT'] - media_params['Wirkungsbeitrag']) / 1
    media_params.fillna(0, inplace=True)
    return media_params


class Magic:

    def __init__(self, media_flighting, media_params, media=MEDIA_LABELS):
        self.media = media                      # media labels
        self.media_flighting = media_flighting  # flighting plan for each media
        self.media_params = media_params        # dict, where each element is a dataframe with separate scenario
        self.media_params_optimized = self.media_params.copy()  # results of optimization

    # Update parameters of each scenario:
    def update_params_for_each_scenario(self, hyperbel_params=None, ots_params=None):
        """update hyperbel and ots parameters in each scenario"""

        for szenario in self.media_params:
            if hyperbel_params is not None:
                assert (len(hyperbel_params) == len(self.media))
                self.media_params[szenario]['Hyperbel'] = hyperbel_params  # update hyperbel params
            if ots_params is not None:
                assert (len(ots_params) == len(self.media))
                self.media_params[szenario]['OTS-beta'] = ots_params  # update ots-params

    def get_optimal_coefs_for_each_scenario(self, normalize=True):
        """for each scenario add a new column with optimal coefficients"""
        for sz in self.media_params:
            opt_coefs = optimized_roi_plus(scenario=self.media_params[sz],
                                           media_flighting=self.media_flighting,
                                           normalize=normalize)
            # update scenario:
            self.media_params_optimized[sz] = opt_coefs


def objective(params, x):
    """define objective function - "distance" between impact-coefficients"""
    x.update_params_for_each_scenario(ots_params=params[:int(len(params)/2)],
                                      hyperbel_params=params[int(len(params)/2):])
    x.get_optimal_coefs_for_each_scenario(normalize=True)
    szenarien_opt = x.media_params_optimized
    coefficients_df = pd.DataFrame()

    for szenario in szenarien_opt:
        coefs = szenarien_opt[szenario]['Koeffizient_opt']
        coefficients_df = coefficients_df.append(coefs)
    metric = 0
    for media in coefficients_df.columns:
        metric += (max(coefficients_df[media]) - min(coefficients_df[media])) ** 2
    return metric


@st.cache
def run_optimization(media_flighting, scenarios_dfs, MEDIA_LABELS_SEL, b_ots, b_hyp, tolerance):
    x = Magic(media_flighting=media_flighting, media_params=scenarios_dfs, media=MEDIA_LABELS_SEL)

    # Load initial OTS-betas and hyperbel-params:
    ots_betas_scalar = x.media_params["szenario_1"]["OTS-beta"].values
    hyp_params_scalar = x.media_params["szenario_1"]["Hyperbel"].values

    # initial parameters (ots und hyperbel)
    params_initial = list(ots_betas_scalar) + list(hyp_params_scalar)

    # Define start values and constraints:
    bnds_ots = [list(b_ots) for _ in ots_betas_scalar]
    bnds_hyp = [list(b_hyp) for _ in hyp_params_scalar]
    bnds = bnds_ots + bnds_hyp

    tolerance = 1 / ((10) ** tolerance)

    ts = time.time()
    sol = minimize(objective,
                   x0=params_initial,
                   args=(x),
                   bounds=bnds,
                   method="trust-constr",
                   tol=tolerance,
                   options={'disp': False})

    exec_time = round((time.time() - ts) / 60, 1)
    return x, exec_time


# User interface
st.title('Magic File')
st.sidebar.subheader("Load file with scenarios")

st.set_option('deprecation.showfileUploaderEncoding', False)
excel = st.sidebar.file_uploader("Choose an excel file", type="xlsx")

if excel is not None:
    excel = pd.read_excel(excel, None)
    st.sidebar.subheader("Select media")
    # media for further considering:
    MEDIA_LABELS_SEL = st.sidebar.multiselect("Media", MEDIA_LABELS, default=MEDIA_LABELS)
    # Flight plan:
    media_flighting = excel["Flight_Plan"][MEDIA_LABELS_SEL]

    if st.sidebar.checkbox('Show flighting plan'):
        st.subheader("Flighting plan:")
        st.dataframe(media_flighting)

    count_flighting_weeks = pd.DataFrame({
        "Media": media_flighting.columns,
        "Anzahl_KW": media_flighting.sum()}, index=None)

    scenarios = excel.keys()  # see all sheet names
    scenarios_dfs = {}  # dictionary for storing DataFrames with media-scenarios
    just_scenarios = []
    for scenario in scenarios:
        if scenario != "Flight_Plan":
            just_scenarios.append(scenario)
            df = excel[scenario]
            df = df[df['Media'].isin(MEDIA_LABELS_SEL)]
            df['Effectiver_CpG'] = df['CpG'] / df['Format']
            df['Effectiver_CpG_OT'] = df['CpG_OT'] / df['Format']
            df = df.set_index("Media")
            # add flighting weeks counts to each scenario:
            df = pd.merge(df, count_flighting_weeks, how="left", on="Media").fillna(0)
            df['Budget_pro_Woche'] = df['Budget'] / df['Anzahl_KW']
            df['Budget_pro_Woche_OT'] = np.where(df['Budget'] != 0, (df['Budget'] + 1) / df['Anzahl_KW'], 0)
            df['Koeffizient'] = 1  # set initial coef value
            scenarios_dfs[scenario] = df

    if st.sidebar.checkbox('Show scenarios'):
        columns_to_show = ['Media', 'CpG', 'CpG_OT', 'Format', 'Adstock', 'Hyperbel',
                           'OTS-beta', 'OTS_Konstante', 'OTS_Steigung', 'Budget']
        for scenario in just_scenarios:
            scenario_name_pretty = scenario.replace("_", " ").capitalize()
            st.subheader(scenario_name_pretty)
            st.dataframe(scenarios_dfs[scenario][columns_to_show])

    # Define start values and constraints:
    b_ots = st.sidebar.slider("Bounds for OTS-Beta", 1.5, 25.0, [1.5, 25.0], 0.5)  # bounds for ots-beta
    b_hyp = st.sidebar.slider("Bounds for Hyperbel", 0.05, 1.0, [0.05, 1.0], 0.05)  # bounds for hyperbel
    tolerance = st.sidebar.slider("Accuracy (magnitude)", 1, 5, 2, 1)

    if st.checkbox('Start optimization'):

        x, exec_time = run_optimization(media_flighting, scenarios_dfs, MEDIA_LABELS_SEL, b_ots, b_hyp, tolerance)
        st.write("Execution time: ", exec_time, "min")

        st.subheader('Results:')
        columns_result = ['Media', 'Adstock', 'Hyperbel', 'OTS-beta', 'Koeffizient_opt']

        df_list = []
        for scenario in just_scenarios:
            tmp_df = x.media_params_optimized[scenario][columns_result].copy()
            tmp_df = tmp_df.rename(columns={'Koeffizient_opt': 'Koef_' + scenario})
            df_list.append(tmp_df)

        results = reduce(lambda left, right: pd.merge(left, right, on=columns_result[:4], how='left'), df_list)
        st.dataframe(results)

        if st.button('Download csv'):
            results.to_csv('optimized_coefficients.csv', index=False, sep=";")
            st.balloons()
            st.markdown("Done! :sunglasses:")

# TODO:

# 0. cache for load           +
# 1. Filter media             +
# 2. Import xlsx-file         +
# 3. Export results in csv    +
# 4. Zero and non-zero media?
# 5. coefficients for media outside media-mix: what is minimal budget and what is logic?
# 6. BI-code for optimal budget: testing without frontend
