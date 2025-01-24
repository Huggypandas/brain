import os
import json
import time
from time import sleep
from os.path import expanduser
from typing import List, Tuple, Dict

import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import numpy as np
from loguru import logger

# 配置文件路径
CREDENTIALS_FILE = expanduser('~/Documents/MyInformation/brain_credentials.txt')
CHECKPOINT_FILE = "alpha_checkpoint.txt"
LOG_FILE = "my_log_file.log"

# API 端点
AUTH_URL = 'https://api.worldquantbrain.com/authentication'
DATAFIELDS_URL = 'https://api.worldquantbrain.com/data-fields'
SIMULATIONS_URL = 'https://api.worldquantbrain.com/simulations'

# 其他常量
ALPHA_FAIL_ATTEMPT_TOLERANCE = 15
DATASET_ID_FUNDAMENTAL6 = 'fundamental6'
LIMIT = 50  # 每页请求数量


def sign_in(user_name: str, password: str) -> requests.Session:
    """
    通过 HTTP Basic Authentication 认证并返回一个 Session 对象。

    Args:
        user_name (str): 用户名。
        password (str): 密码。

    Returns:
        requests.Session: 认证后的 Session 对象。
    """
    sess = requests.Session()
    sess.auth = HTTPBasicAuth(user_name, password)

    response = sess.post(AUTH_URL)
    response.raise_for_status()  # 如果认证失败，抛出异常

    logger.info("Successfully signed in.")
    return sess


def get_datafields(
    session: requests.Session,
    search_scope: Dict[str, str],
    dataset_id: str = '',
    search: str = ''
) -> pd.DataFrame:
    """
    从 API 获取数据字段并返回为 DataFrame。

    Args:
        session (requests.Session): 已认证的 Session 对象。
        search_scope (Dict[str, str]): 搜索范围参数。
        dataset_id (str, optional): 数据集 ID。默认为空字符串。
        search (str, optional): 搜索关键词。默认为空字符串。

    Returns:
        pd.DataFrame: 包含数据字段的 DataFrame。
    """
    instrument_type = search_scope.get('instrumentType', '')
    region = search_scope.get('region', '')
    delay = search_scope.get('delay', '')
    universe = search_scope.get('universe', '')

    if search:
        url_template = (
            f"{DATAFIELDS_URL}?"
            f"instrumentType={instrument_type}"
            f"&region={region}&delay={delay}&universe={universe}"
            f"&limit={LIMIT}"
            f"&search={search}"
            f"&offset={{x}}"
        )
        count = 100  # 假设有100条结果
    else:
        url_template = (
            f"{DATAFIELDS_URL}?"
            f"instrumentType={instrument_type}"
            f"&region={region}&delay={delay}&universe={universe}"
            f"&dataset.id={dataset_id}&limit={LIMIT}"
            f"&offset={{x}}"
        )
        initial_response = session.get(url_template.format(x=0))
        initial_response.raise_for_status()
        count = initial_response.json().get('count', 0)

    datafields_list = []
    for offset in range(0, count, LIMIT):
        response = session.get(url_template.format(x=offset))
        response.raise_for_status()
        datafields_list.append(response.json().get('results', []))

    # 展平列表并转换为 DataFrame
    datafields_flat = [item for sublist in datafields_list for item in sublist]
    datafields_df = pd.DataFrame(datafields_flat)
    logger.info(f"Fetched {len(datafields_df)} data fields.")
    return datafields_df


def generate_alpha_expressions(
    group_compare_op: List[str],
    ts_compare_op: List[str],
    company_fundamentals: List[str],
    days: List[int],
    group: List[str]
) -> List[str]:
    """
    生成 Alpha 表达式列表。

    Args:
        group_compare_op (List[str]): 分组比较操作列表。
        ts_compare_op (List[str]): 时间序列比较操作列表。
        company_fundamentals (List[str]): 公司基本面字段列表。
        days (List[int]): 时间窗口列表。
        group (List[str]): 分组方式列表。

    Returns:
        List[str]: 生成的 Alpha 表达式列表。
    """
    alpha_expressions = [
        f"{gco}({tco}({cf},{d}), {grp})"
        for gco in group_compare_op
        for tco in ts_compare_op
        for cf in company_fundamentals
        for d in days
        for grp in group
    ]
    logger.info(f"Generated {len(alpha_expressions)} alpha expressions.")
    return alpha_expressions


def generate_alpha_expressions_2(datafields_list_fundamental6):
    alpha_list = []
    group = [
        "market",
        "industry",
        "subindustry",
        "sector"
    ]
    for datafield in datafields_list_fundamental6:
        for g in group:
            # 根据当前 datafield 构造 Alpha 表达式
            alpha_expression = f"group_rank({datafield}/cap, {g})"
            alpha_list.append(alpha_expression)
    
    logger.info(f"there are {len(alpha_list)} Alphas to simulate")
    return alpha_list


def save_checkpoint(index: int, checkpoint_file: str = CHECKPOINT_FILE) -> None:
    """
    保存当前处理的 Alpha 索引到检查点文件。

    Args:
        index (int): 当前处理的 Alpha 索引。
        checkpoint_file (str, optional): 检查点文件路径。默认为 'alpha_checkpoint.txt'。
    """
    with open(checkpoint_file, "w") as f:
        f.write(str(index))
    logger.debug(f"Checkpoint saved at index {index}.")


def load_checkpoint(checkpoint_file: str = CHECKPOINT_FILE) -> int:
    """
    从检查点文件加载当前处理的 Alpha 索引。

    Args:
        checkpoint_file (str, optional): 检查点文件路径。默认为 'alpha_checkpoint.txt'。

    Returns:
        int: 加载的 Alpha 索引。如果文件不存在，返回0。
    """
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            start_index = int(f.read().strip())
        logger.info(f"Loaded checkpoint. Resuming from index {start_index}.")
        return start_index
    logger.info("No checkpoint found. Starting from index 0.")
    return 0


def process_alpha(
    session: requests.Session,
    alpha: str,
    index: int
) -> bool:
    """
    处理单个 Alpha 表达式，发送模拟请求并处理响应。

    Args:
        session (requests.Session): 已认证的 Session 对象。
        alpha (str): Alpha 表达式。
        index (int): 当前 Alpha 的索引。

    Returns:
        bool: 如果处理成功，返回 True；否则返回 False。
    """
    simulation_data = {
        'type': 'REGULAR',
        'settings': {
            'instrumentType': 'EQUITY',
            'region': 'USA',
            'universe': 'TOP3000',
            'delay': 1,
            'decay': 0,
            'neutralization': 'SUBINDUSTRY',
            'truncation': 0.08,
            'pasteurization': 'ON',
            'unitHandling': 'VERIFY',
            'nanHandling': 'ON',
            'language': 'FASTEXPR',
            'visualization': False,
        },
        'regular': alpha
    }

    try:
        response = session.post(SIMULATIONS_URL, json=simulation_data)
        response.raise_for_status()
        sim_progress_url = response.headers.get('Location', '')
        if sim_progress_url:
            logger.info(f"Alpha index {index}: Simulation URL - {sim_progress_url}")
            return True
        else:
            logger.warning(f"Alpha index {index}: No Location header found.")
            return False
    except requests.RequestException as e:
        logger.error(f"Alpha index {index}: Request failed with error: {e}")
        return False


def main():
    # 配置日志
    logger.add(LOG_FILE, format="{time} {level} {message}", level="INFO")

    # 读取凭证
    if not os.path.isfile(CREDENTIALS_FILE):
        logger.error(f"Credentials file not found at {CREDENTIALS_FILE}. Exiting.")
        return

    with open(CREDENTIALS_FILE) as f:
        credentials = json.load(f)

    if not isinstance(credentials, list) or len(credentials) != 2:
        logger.error("Credentials file format is incorrect. It should be a list of [username, password]. Exiting.")
        return

    user_name, password = credentials

    # 认证
    try:
        sess = sign_in(user_name, password)
    except requests.RequestException as e:
        logger.error(f"Failed to sign in: {e}. Exiting.")
        return

    # 定义搜索范围
    search_scope = {
        'region': 'USA',
        'delay': '1',
        'universe': 'TOP3000',
        'instrumentType': 'EQUITY'
    }

    # 获取数据字段
    try:
        fundamental6_df = get_datafields(
            session=sess,
            search_scope=search_scope,
            dataset_id=DATASET_ID_FUNDAMENTAL6
        )
    except requests.RequestException as e:
        logger.error(f"Failed to fetch data fields: {e}. Exiting.")
        return

    # 过滤 type 为 MATRIX 的行
    fundamental6_df = fundamental6_df[fundamental6_df['type'] == "MATRIX"]
    datafields_list_fundamental6 = fundamental6_df['id'].tolist()

    # 定义比较操作列表
    group_compare_op = ["group_rank", "group_zscore", "group_neutralize"]  # ...
    ts_compare_op = ["ts_rank", "ts_zscore", "ts_av_diff"]

    # 时间窗口列表
    days = [600, 200]

    # 分组方式列表
    group = [
        "market",
        "industry",
        "subindustry",
        "sector",
        "densify(pv13_h_fl_sector)"  # 示例：自定义的“densify”分组表达式
    ]

    # 生成 Alpha 表达式
    alpha_expressions = generate_alpha_expressions_2(
        datafields_list_fundamental6
    )

    # 加载检查点
    start_index = load_checkpoint()

    # 处理 Alpha 表达式
    for i, alpha in enumerate(alpha_expressions[start_index:], start=start_index):
        logger.info(f"Processing alpha index {i}, alpha: {alpha}")

        keep_trying = True
        failure_count = 0

        while keep_trying:
            success = process_alpha(session=sess, alpha=alpha, index=i)
            if success:
                keep_trying = False
            else:
                logger.error(f"Alpha index {i}: Failed to process. Retrying after 15 seconds.")
                sleep(15)
                failure_count += 1

                if failure_count > ALPHA_FAIL_ATTEMPT_TOLERANCE:
                    try:
                        sess = sign_in(user_name, password)
                        logger.info(f"Alpha index {i}: Re-signed in after {failure_count} failures.")
                    except requests.RequestException as e:
                        logger.error(f"Alpha index {i}: Re-sign in failed with error: {e}. Skipping this alpha.")
                        break  # 跳过当前 alpha，进入下一个

                    failure_count = 0
                    keep_trying = False  # 跳过当前 alpha

        # 保存检查点
        save_checkpoint(i + 1)

    logger.info("All alpha expressions processed or attempted.")


if __name__ == '__main__':
    main()