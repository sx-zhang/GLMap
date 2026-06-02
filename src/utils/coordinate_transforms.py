import numpy as np


def xRyUzB_position_to_xFyLzU_position(xRyUzB_position: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X right, Y up, Z back
    Output coordinate system: X forward, Y left, Z up
    xRyUzB is the default coordinate system of Habitat simulator, while xFyLzU is the common base coordinate system in robotics.
    """
    xFyLzU_position = xRyUzB_position.copy()
    xFyLzU_position[0] = -xRyUzB_position[2]
    xFyLzU_position[1] = -xRyUzB_position[0]
    xFyLzU_position[2] = xRyUzB_position[1]
    return xFyLzU_position


def xRyUzB_pcd_to_xFyLzU_pcd(xRyUzB_pcd: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X right, Y up, Z back
    Output coordinate system: X forward, Y left, Z up
    Convert point cloud from xRyUzB coordinate system to xFyLzU coordinate system
    """
    xFyLzU_pcd = xRyUzB_pcd.copy()
    xFyLzU_pcd[:, 0] = -xRyUzB_pcd[:, 2]
    xFyLzU_pcd[:, 1] = -xRyUzB_pcd[:, 0]
    xFyLzU_pcd[:, 2] = xRyUzB_pcd[:, 1]
    return xFyLzU_pcd


def xFyLzU_position_to_xRyUzB_position(xFyLzU_position: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X forward, Y left, Z up
    Output coordinate system: X right, Y up, Z back
    xFyLzU is the common base coordinate system in robotics, while xRyUzB is the default coordinate system of Habitat simulator.
    """
    xRyUzB_position = xFyLzU_position.copy()
    xRyUzB_position[0] = -xFyLzU_position[1]
    xRyUzB_position[1] = xFyLzU_position[2]
    xRyUzB_position[2] = -xFyLzU_position[0]
    return xRyUzB_position


def xFyLzU_pcd_to_xRyUzB_pcd(xFyLzU_pcd: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X forward, Y left, Z up
    Output coordinate system: X right, Y up, Z back
    Convert point cloud from xFyLzU coordinate system to xRyUzB coordinate system
    """
    xRyUzB_pcd = xFyLzU_pcd.copy()
    xRyUzB_pcd[:, 0] = -xFyLzU_pcd[:, 1]
    xRyUzB_pcd[:, 1] = xFyLzU_pcd[:, 2]
    xRyUzB_pcd[:, 2] = -xFyLzU_pcd[:, 0]
    return xRyUzB_pcd


def xFyLzU_position_to_xRyDzF_position(xFyLzU_position: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X forward, Y left, Z up
    Output coordinate system: X right, Y down, Z forward
    xFyLzU is the common base coordinate system in robotics, xRyDzF is the coordinate system of COLMAP/OpenCV.
    """
    xRyDzF_position = xFyLzU_position.copy()
    xRyDzF_position[0] = -xFyLzU_position[1]
    xRyDzF_position[1] = -xFyLzU_position[2]
    xRyDzF_position[2] = xFyLzU_position[0]
    return xRyDzF_position


def xFyLzU_pcd_to_xRyDzF_pcd(xFyLzU_pcd: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X forward, Y left, Z up
    Output coordinate system: X right, Y down, Z forward
    Convert point cloud from xFyLzU coordinate system to xRyDzF (COLMAP/OpenCV) coordinate system
    """
    xRyDzF_pcd = xFyLzU_pcd.copy()
    xRyDzF_pcd[:, 0] = -xFyLzU_pcd[:, 1]
    xRyDzF_pcd[:, 1] = -xFyLzU_pcd[:, 2]
    xRyDzF_pcd[:, 2] = xFyLzU_pcd[:, 0]
    return xRyDzF_pcd


def xRyDzF_pcd_to_xFyLzU_pcd(xRyDzF_pcd: np.ndarray) -> np.ndarray:
    """
    Input coordinate system: X right, Y down, Z forward
    Output coordinate system: X forward, Y left, Z up
    Convert point cloud from xRyDzF (COLMAP/OpenCV) coordinate system to xFyLzU coordinate system
    """
    xFyLzU_pcd = xRyDzF_pcd.copy()
    xFyLzU_pcd[:, 0] = xRyDzF_pcd[:, 2]
    xFyLzU_pcd[:, 1] = -xRyDzF_pcd[:, 0]
    xFyLzU_pcd[:, 2] = -xRyDzF_pcd[:, 1]
    return xFyLzU_pcd


def map_rBcR_to_map_rLcF(
    map_row: int,
    map_col: int,
    map_row_for_position_zero_rBcR: int,
    map_col_for_position_zero_rBcR: int,
    map_row_for_position_zero_rLcF: int,
    map_col_for_position_zero_rLcF: int,
) -> tuple[int, int]:
    """
    Convert rBcR map coordinates to rLcF map coordinates
    rBcR: row corresponds to backward, col corresponds to right
    rLcF: row corresponds to left, col corresponds to forward

    Args:
        map_row: Row coordinate in rBcR map
        map_col: Column coordinate in rBcR map
        map_row_for_position_zero_rBcR: Row origin of rBcR map
        map_col_for_position_zero_rBcR: Column origin of rBcR map
        map_row_for_position_zero_rLcF: Row origin of rLcF map
        map_col_for_position_zero_rLcF: Column origin of rLcF map

    Returns:
        (rLcF_row, rLcF_col): Row and column coordinates in rLcF map
    """
    # rBcR: backward = (map_row - map_row_for_position_zero_rBcR), right = (map_col - map_col_for_position_zero_rBcR)
    # rLcF: left = backward, forward = -right
    # rLcF_row = left + map_row_for_position_zero_rLcF, rLcF_col = forward + map_col_for_position_zero_rLcF

    rLcF_row = (
        map_row - map_row_for_position_zero_rBcR
    ) + map_row_for_position_zero_rLcF
    rLcF_col = (
        -(map_col - map_col_for_position_zero_rBcR) + map_col_for_position_zero_rLcF
    )

    return int(rLcF_row), int(rLcF_col)


def map_rBcR_to_map_rLcF_array(
    map_row_col: np.ndarray,
    map_row_for_position_zero_rBcR: int,
    map_col_for_position_zero_rBcR: int,
    map_row_for_position_zero_rLcF: int,
    map_col_for_position_zero_rLcF: int,
) -> np.ndarray:
    """
    Convert rBcR map coordinate array to rLcF map coordinate array
    rBcR: row corresponds to backward, col corresponds to right
    rLcF: row corresponds to left, col corresponds to forward

    Args:
        map_row_col: Row and column coordinate array of rBcR map, shape (N, 2)
        map_row_for_position_zero_rBcR: Row origin of rBcR map
        map_col_for_position_zero_rBcR: Column origin of rBcR map
        map_row_for_position_zero_rLcF: Row origin of rLcF map
        map_col_for_position_zero_rLcF: Column origin of rLcF map

    Returns:
        Row and column coordinate array of rLcF map, shape (N, 2)
    """
    rLcF_row_col = np.zeros_like(map_row_col)
    rLcF_row_col[:, 0] = (
        map_row_col[:, 0] - map_row_for_position_zero_rBcR
    ) + map_row_for_position_zero_rLcF
    rLcF_row_col[:, 1] = (
        -(map_row_col[:, 1] - map_col_for_position_zero_rBcR)
        + map_col_for_position_zero_rLcF
    )

    return rLcF_row_col.astype(int)


def map_rLcF_to_map_rBcR(
    map_row: int,
    map_col: int,
    map_row_for_position_zero_rLcF: int,
    map_col_for_position_zero_rLcF: int,
    map_row_for_position_zero_rBcR: int,
    map_col_for_position_zero_rBcR: int,
) -> tuple[int, int]:
    """
    Convert rLcF map coordinates to rBcR map coordinates
    rLcF: row corresponds to left, col corresponds to forward
    rBcR: row corresponds to backward, col corresponds to right

    Args:
        map_row: Row coordinate in rLcF map
        map_col: Column coordinate in rLcF map
        map_row_for_position_zero_rLcF: Row origin of rLcF map
        map_col_for_position_zero_rLcF: Column origin of rLcF map
        map_row_for_position_zero_rBcR: Row origin of rBcR map
        map_col_for_position_zero_rBcR: Column origin of rBcR map

    Returns:
        (rBcR_row, rBcR_col): Row and column coordinates in rBcR map
    """
    # rLcF: left = (map_row - map_row_for_position_zero_rLcF), forward = (map_col - map_col_for_position_zero_rLcF)
    # rBcR: backward = left, right = -forward
    # rBcR_row = backward + map_row_for_position_zero_rBcR, rBcR_col = right + map_col_for_position_zero_rBcR

    rBcR_row = (
        map_row - map_row_for_position_zero_rLcF
    ) + map_row_for_position_zero_rBcR
    rBcR_col = (
        -(map_col - map_col_for_position_zero_rLcF) + map_col_for_position_zero_rBcR
    )

    return int(rBcR_row), int(rBcR_col)


def map_rLcF_to_map_rBcR_array(
    map_row_col: np.ndarray,
    map_row_for_position_zero_rLcF: int,
    map_col_for_position_zero_rLcF: int,
    map_row_for_position_zero_rBcR: int,
    map_col_for_position_zero_rBcR: int,
) -> np.ndarray:
    """
    Convert rLcF map coordinate array to rBcR map coordinate array
    rLcF: row corresponds to left, col corresponds to forward
    rBcR: row corresponds to backward, col corresponds to right

    Args:
        map_row_col: Row and column coordinate array of rLcF map, shape (N, 2)
        map_row_for_position_zero_rLcF: Row origin of rLcF map
        map_col_for_position_zero_rLcF: Column origin of rLcF map
        map_row_for_position_zero_rBcR: Row origin of rBcR map
        map_col_for_position_zero_rBcR: Column origin of rBcR map

    Returns:
        Row and column coordinate array of rBcR map, shape (N, 2)
    """
    rBcR_row_col = np.zeros_like(map_row_col)
    rBcR_row_col[:, 0] = (
        map_row_col[:, 0] - map_row_for_position_zero_rLcF
    ) + map_row_for_position_zero_rBcR
    rBcR_row_col[:, 1] = (
        -(map_row_col[:, 1] - map_col_for_position_zero_rLcF)
        + map_col_for_position_zero_rBcR
    )

    return rBcR_row_col.astype(int)
