"""Script tam thoi: sua cac dong print tieng Viet cuoi app.py thanh ASCII"""
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Tim vi tri dong if __name__ == '__main__':
main_idx = None
for i, line in enumerate(lines):
    if "if __name__" in line and "'__main__'" in line or '"__main__"' in line:
        main_idx = i
        break

if main_idx is not None:
    # Giu lai phan code truoc
    new_lines = lines[:main_idx]
    new_lines.append('\nif __name__ == "__main__":\n')
    new_lines.append('    print("\\n" + "="*50)\n')
    new_lines.append('    print("WEB SERVER STARTED - HTTPS MODE")\n')
    new_lines.append('    print("Run ipconfig to find your IPv4. E.g: 192.168.1.xxx")\n')
    new_lines.append('    print("On your PHONE use Chrome and go to: https://<your_ipv4>:5000")\n')
    new_lines.append('    print("If browser warns Not Secure: click Advanced then Continue")\n')
    new_lines.append('    print("="*50 + "\\n")\n')
    new_lines.append('    app.run(host="0.0.0.0", port=5000, debug=False, ssl_context="adhoc")\n')

    with open('app.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Done - app.py updated successfully")
else:
    print("ERROR: Could not find __main__ block")
