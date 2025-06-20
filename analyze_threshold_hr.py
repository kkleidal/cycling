import argparse
from datetime import timedelta
import gpxpy

def main():
    parser = argparse.ArgumentParser(description='Analyze threshold heart rate')
    parser.add_argument('gpx_file', type=argparse.FileType('r'), help='Input GPX file')

    args = parser.parse_args()
    gpx = gpxpy.parse(args.gpx_file)
    times = []
    hrs = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                hr = point.extensions[0].find('{http://www.garmin.com/xmlschemas/TrackPointExtension/v1}hr').text
                time = point.time
                times.append(time)
                hrs.append(int(hr))
    last_time = times[-1]
    cutoff = last_time - timedelta(minutes=20)
    mask = [time > cutoff for time in times]
    times = [time for time in times if time > cutoff]
    hrs = [hr for hr, m in zip(hrs, mask) if m]
    total = 0
    total_time = 0
    for i in range(1, len(times)):
        dt = (times[i] - times[i-1]).total_seconds()
        total += (hrs[i] + hrs[i-1]) / 2 * dt
        total_time += dt
    print(total / total_time)
    print(0.9 * (total / total_time))


if __name__ == '__main__':
    main()