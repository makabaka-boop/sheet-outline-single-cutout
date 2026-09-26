# 钣金外轮廓重建服务

纯后端服务：接收测量仪导出的乱序轴对齐边段，重建端点邻接关系，确认它们能组成
一条无自交的闭合外轮廓后，返回稳定、可复算的顺时针轮廓。全程整数运算，无浮点误差。

## 运行

```bash
docker compose up --build        # api 服务监听 :8000
```

本地开发（Python 3.12）：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest                          # 整数几何夹具 + 打乱不变性测试
```

## API

### `POST /reconstruct`

请求体：

```json
{
  "segments": [
    {"id": "e1", "a": [0, 0], "b": [2, 0]},
    {"id": "e2", "a": [2, 0], "b": [2, 2]}
  ]
}
```

- 4 至 500 条线段，`id` 为唯一非空字符串；
- 端点为整数坐标，`|x|, |y| <= 10^6`；
- 只接受水平 / 垂直线段。

**200 响应**（从字典序最小顶点出发的顺时针轮廓）：

```json
{
  "vertices": [[0, 0], [0, 2], [2, 2], [2, 0]],
  "edge_ids": ["e2", "e3", "e1", "e4"],
  "doubled_area": 8,
  "perimeter": 8
}
```

- `vertices[i]` 沿 `edge_ids[i]` 走到 `vertices[i+1]`（末边回到起点）；
- `doubled_area` 为整数鞋带公式结果的两倍面积（恒为整数，避免浮点）；
- 同一批边任意打乱顺序，响应逐字节一致。

**422 响应**（整批拒绝，携带确定性证据）：

```json
{"detail": {"code": "BAD_DEGREE", "message": "...", "witness": {"point": [2, 2]}}}
```

输入非法（先检）：

| code | 含义 |
| --- | --- |
| `INVALID_PAYLOAD` / `INVALID_SEGMENT_COUNT` / `INVALID_SEGMENT_ID` / `DUPLICATE_SEGMENT_ID` / `INVALID_POINT` / `COORDINATE_OUT_OF_RANGE` | 结构、数量、id、坐标非法 |
| `ZERO_LENGTH_SEGMENT` / `NON_AXIS_ALIGNED` | 零长度 / 非轴对齐线段 |
| `DUPLICATE_EDGE` / `PARTIAL_OVERLAP` | 重复边 / 共线部分重叠（witness 含两条线段 id） |

拓扑重建失败（按此顺序，前者优先）：

| code | 含义 | witness |
| --- | --- | --- |
| `BAD_DEGREE` | 某端点度数 ≠ 2（角点接触、T 型接触、断头） | 字典序最小的坏端点 `{"point": [x, y]}` |
| `DISCONNECTED` | 边集分成多个环 | 主分量外最小线段 id `{"segment_id": ...}` |
| `SELF_INTERSECTION` | 非相邻边相交或接触 | 相交边中最小 id `{"segment_id": ...}` |

主分量定义为包含字典序最小端点的分量。所有 witness 都按字典序 / id 升序选取，
因此同一批乱序输入的拒绝证据也完全稳定。

## 结构

```
app/geometry.py   纯整数几何核心：校验、度数/连通性/自交检查、遍历与测量
app/main.py       FastAPI 入口，统一 422 错误格式
tests/            pytest：凹多边形、180° 共线顶点、角点接触、T 型接触、
                  自交单环、多环、重叠/重复边、坐标边界、500 条边上界、
                  以及所有夹具的打乱不变性与独立复算（鞋带公式）校验
```

## 关键决策

- **顺时针**采用标准笛卡尔取向（鞋带符号为负）；若遍历得到逆时针序列则整体反转，
  起点保持字典序最小顶点不变。
- 起点处若两条出边都朝“内角允许”的方向（仅 180° 共线顶点可能），按最小线段 id
  确定性选择，方向最终由鞋带符号统一校正。
- 共线边只允许相离或共享恰好一个端点（相邻）；任何正长度共享区间都使整批以
  422 拒绝，因为此类输入对重建而言有歧义。
