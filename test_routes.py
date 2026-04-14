from app import app

route = ['/', '/analysis', '/qr_upload', '/transaction', '/get_chart_data', '/nonexistent']

with app.test_client() as c:
    for r in route:
        resp = c.get(r)
        print(r, resp.status_code)
        data = resp.get_data(as_text=True)
        print(data[:500])
        print('----')

