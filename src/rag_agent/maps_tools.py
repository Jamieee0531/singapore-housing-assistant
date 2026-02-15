"""
Google Maps 工具模块 - 提供位置、通勤和周边搜索功能

三个工具:
1. get_commute_info - 计算通勤距离和时间
2. get_directions - 获取详细路线指引
3. search_nearby - 搜索附近设施
"""

from typing import List
from langchain_core.tools import tool
import googlemaps
from src.config import MAPS_SEARCH_RADIUS, MAPS_MAX_RESULTS


# 新加坡常见地点缩写映射
SINGAPORE_LOCATIONS = {
    "NUS": "National University of Singapore",
    "NTU": "Nanyang Technological University",
    "SMU": "Singapore Management University",
    "SUTD": "Singapore University of Technology and Design",
    "CBD": "Central Business District Singapore",
    "MBS": "Marina Bay Sands Singapore",
    "Orchard": "Orchard Road Singapore",
}


def _normalize_location(location: str) -> str:
    """
    规范化地点名称，添加新加坡上下文

    - 识别 NUS/NTU 等缩写
    - 自动添加 ", Singapore" 后缀
    """
    # 检查是否是已知缩写
    upper_loc = location.upper().strip()
    if upper_loc in SINGAPORE_LOCATIONS:
        return SINGAPORE_LOCATIONS[upper_loc]

    # 如果没有包含 Singapore，添加后缀
    if "singapore" not in location.lower():
        return f"{location}, Singapore"

    return location


class MapsToolFactory:
    """
    Google Maps 工具工厂类

    使用方法:
        factory = MapsToolFactory(api_key)
        tools = factory.create_tools()
    """

    def __init__(self, api_key: str):
        """
        初始化 Maps 工具工厂

        Args:
            api_key: Google Maps API Key
        """
        self.client = googlemaps.Client(key=api_key)

    def create_tools(self) -> List:
        """创建并返回所有 Maps 工具"""

        client = self.client

        @tool
        def get_commute_info(origin: str, destination: str) -> str:
            """
            计算两地之间的通勤距离和时间。

            适用场景：
            - "从金文泰到NUS要多久？"
            - "Clementi 到 CBD 多远？"
            - "住哪里离学校近？"

            Args:
                origin: 出发地点（如 "Clementi", "金文泰"）
                destination: 目的地点（如 "NUS", "CBD"）

            Returns:
                包含距离和时间的通勤信息
            """
            try:
                # 规范化地点名称
                origin_normalized = _normalize_location(origin)
                destination_normalized = _normalize_location(destination)

                # 获取公共交通路线
                result = client.distance_matrix(
                    origins=[origin_normalized],
                    destinations=[destination_normalized],
                    mode="transit",
                    region="sg"
                )

                if result["rows"][0]["elements"][0]["status"] != "OK":
                    return f"无法计算从 {origin} 到 {destination} 的路线，请检查地点名称是否正确。"

                element = result["rows"][0]["elements"][0]
                distance = element["distance"]["text"]
                duration = element["duration"]["text"]

                # 同时获取驾车时间作为参考
                driving_result = client.distance_matrix(
                    origins=[origin_normalized],
                    destinations=[destination_normalized],
                    mode="driving",
                    region="sg"
                )

                driving_duration = ""
                if driving_result["rows"][0]["elements"][0]["status"] == "OK":
                    driving_duration = driving_result["rows"][0]["elements"][0]["duration"]["text"]

                response = f"""从 {origin} 到 {destination} 的通勤信息：

公共交通：
  - 距离: {distance}
  - 时间: {duration}

驾车参考: {driving_duration if driving_duration else '无法获取'}

提示: 新加坡公共交通非常发达，MRT是最常用的通勤方式。"""

                return response

            except Exception as e:
                return f"获取通勤信息时出错: {str(e)}。请确保地点名称正确。"

        @tool
        def get_directions(origin: str, destination: str) -> str:
            """
            获取从出发地到目的地的详细路线指引。

            适用场景：
            - "从金文泰到学校怎么走？"
            - "怎么从 Jurong East 去 NUS？"

            Args:
                origin: 出发地点
                destination: 目的地点

            Returns:
                详细的路线步骤说明
            """
            try:
                origin_normalized = _normalize_location(origin)
                destination_normalized = _normalize_location(destination)

                directions = client.directions(
                    origin=origin_normalized,
                    destination=destination_normalized,
                    mode="transit",
                    region="sg"
                )

                if not directions:
                    return f"无法获取从 {origin} 到 {destination} 的路线，请检查地点名称。"

                route = directions[0]
                legs = route["legs"][0]

                total_distance = legs["distance"]["text"]
                total_duration = legs["duration"]["text"]

                steps_text = []
                for i, step in enumerate(legs["steps"], 1):
                    instruction = step["html_instructions"]
                    # 清理 HTML 标签
                    instruction = instruction.replace("<b>", "").replace("</b>", "")
                    instruction = instruction.replace("<div style=\"font-size:0.9em\">", " - ")
                    instruction = instruction.replace("</div>", "")

                    step_distance = step["distance"]["text"]
                    step_duration = step["duration"]["text"]

                    travel_mode = step.get("travel_mode", "")
                    if travel_mode == "TRANSIT":
                        transit_details = step.get("transit_details", {})
                        line = transit_details.get("line", {})
                        line_name = line.get("short_name", line.get("name", ""))
                        if line_name:
                            instruction = f"乘坐 {line_name} - {instruction}"
                    elif travel_mode == "WALKING":
                        instruction = f"步行: {instruction}"

                    steps_text.append(f"{i}. {instruction} ({step_distance}, {step_duration})")

                response = f"""从 {origin} 到 {destination} 的路线：

总距离: {total_distance}
预计时间: {total_duration}

详细步骤:
{chr(10).join(steps_text)}

提示: 可使用 Google Maps 或 Citymapper 应用获取实时导航。"""

                return response

            except Exception as e:
                return f"获取路线时出错: {str(e)}。请确保地点名称正确。"

        @tool
        def search_nearby(location: str, place_type: str = "transit_station") -> str:
            """
            搜索指定地点附近的设施。

            适用场景：
            - "Clementi 附近有什么 MRT？"
            - "这个地方周围有超市吗？"
            - "附近有什么好吃的？"

            Args:
                location: 搜索中心位置（如 "Clementi", "Jurong East"）
                place_type: 设施类型，可选值:
                    - "transit_station" (MRT/公交站，默认)
                    - "supermarket" (超市)
                    - "restaurant" (餐厅)
                    - "shopping_mall" (商场)
                    - "hospital" (医院)
                    - "school" (学校)

            Returns:
                附近设施列表
            """
            try:
                location_normalized = _normalize_location(location)

                # 先地理编码获取坐标
                geocode_result = client.geocode(location_normalized)
                if not geocode_result:
                    return f"无法找到地点: {location}，请检查名称是否正确。"

                lat = geocode_result[0]["geometry"]["location"]["lat"]
                lng = geocode_result[0]["geometry"]["location"]["lng"]

                # 设施类型映射
                type_mapping = {
                    "transit_station": "transit_station",
                    "mrt": "transit_station",
                    "supermarket": "supermarket",
                    "restaurant": "restaurant",
                    "shopping_mall": "shopping_mall",
                    "hospital": "hospital",
                    "school": "school",
                    "gym": "gym",
                    "park": "park",
                }

                search_type = type_mapping.get(place_type.lower(), "transit_station")

                # 搜索附近设施
                places_result = client.places_nearby(
                    location=(lat, lng),
                    radius=MAPS_SEARCH_RADIUS,
                    type=search_type
                )

                if not places_result.get("results"):
                    return f"在 {location} 附近{MAPS_SEARCH_RADIUS}米内未找到 {place_type}。"

                # 格式化结果
                type_names = {
                    "transit_station": "交通站点",
                    "supermarket": "超市",
                    "restaurant": "餐厅",
                    "shopping_mall": "商场",
                    "hospital": "医院",
                    "school": "学校",
                    "gym": "健身房",
                    "park": "公园",
                }

                type_name = type_names.get(search_type, place_type)

                places_list = []
                for place in places_result["results"][:MAPS_MAX_RESULTS]:
                    name = place["name"]
                    rating = place.get("rating", "无评分")
                    vicinity = place.get("vicinity", "")

                    if rating != "无评分":
                        places_list.append(f"  - {name} (评分:{rating})\n    地址: {vicinity}")
                    else:
                        places_list.append(f"  - {name}\n    地址: {vicinity}")

                response = f"""{location} 附近的{type_name}（{MAPS_SEARCH_RADIUS}米范围内）：

{chr(10).join(places_list)}

共找到 {len(places_result['results'])} 个结果，以上为前 {min(MAPS_MAX_RESULTS, len(places_result['results']))} 个。"""

                return response

            except Exception as e:
                return f"搜索附近设施时出错: {str(e)}。请确保地点名称正确。"

        return [get_commute_info, get_directions, search_nearby]
