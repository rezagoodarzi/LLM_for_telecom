"""
ELEX PDF Crawler - Downloads all PDFs from Ericsson Library Explorer
Based on actual ELEX structure analysis
"""

import os
import time
import re
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


class ElexCrawler:
    def __init__(self, base_url: str, download_dir: str = "elex_downloads"):
        self.base_url = base_url
        self.download_dir = os.path.abspath(download_dir)
        self.downloaded_docs = set()
        self.failed_downloads = []
        self.success_count = 0
        
        os.makedirs(self.download_dir, exist_ok=True)
        print(f"[INFO] Downloads will be saved to: {self.download_dir}")

    def sanitize_filename(self, name: str) -> str:
        """Create safe filename from document title"""
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        name = name.strip()[:150]
        return name if name else "document"

    def wait_for_load(self, page, seconds=2):
        """Wait for dynamic content to load"""
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except:
            pass
        time.sleep(seconds)

    def get_all_tree_categories(self, page) -> list:
        """Get all expandable categories from the tree"""
        # JavaScript to get all clickable tree items
        categories = page.evaluate('''() => {
            const items = [];
            // Find all list items in the Contents tree
            const listItems = document.querySelectorAll('#PanelContent li, .eaNetwork-FiltersBlock1 li');
            for (const li of listItems) {
                const clickable = li.querySelector('[onclick], div[style*="cursor"], span[style*="cursor"]');
                const textEl = li.querySelector('span, div');
                if (textEl) {
                    const text = textEl.innerText?.trim();
                    if (text && text.length > 2) {
                        items.push(text);
                    }
                }
            }
            return [...new Set(items)];
        }''')
        return categories or []

    def expand_tree_node(self, page, node_text: str) -> bool:
        """Expand a specific tree node by text"""
        try:
            # Find and click the tree node by text
            result = page.evaluate(f'''(nodeText) => {{
                const elements = document.querySelectorAll('li span, li div');
                for (const el of elements) {{
                    if (el.innerText?.trim() === nodeText) {{
                        el.click();
                        return true;
                    }}
                }}
                return false;
            }}''', node_text)
            self.wait_for_load(page, 1)
            return result
        except:
            return False

    def get_document_links_from_table(self, page) -> list:
        """Get all document links from the current table view"""
        docs = page.evaluate('''() => {
            const docs = [];
            // Find all links in the document table
            const rows = document.querySelectorAll('table tbody tr, .ebTable tbody tr');
            for (const row of rows) {
                const link = row.querySelector('a');
                const titleCell = row.querySelector('td:first-child');
                if (link && titleCell) {
                    const title = titleCell.innerText?.trim();
                    if (title && title.length > 2) {
                        docs.push(title);
                    }
                }
            }
            return docs;
        }''')
        return docs or []

    def click_document_by_title(self, page, title: str) -> bool:
        """Click a document link by its title"""
        try:
            result = page.evaluate(f'''(title) => {{
                const links = document.querySelectorAll('table a, .ebTable a');
                for (const link of links) {{
                    if (link.innerText?.trim() === title) {{
                        link.click();
                        return true;
                    }}
                }}
                return false;
            }}''', title)
            if result:
                self.wait_for_load(page, 3)
            return result
        except:
            return False

    def download_current_pdf(self, page, title: str) -> bool:
        """Download PDF of current document using JavaScript click"""
        if title in self.downloaded_docs:
            print(f"[SKIP] Already downloaded: {title[:50]}")
            return True

        try:
            # Check if pdf_save button exists
            has_pdf = page.evaluate('''() => {
                const btn = document.getElementById('pdf_save');
                return btn !== null;
            }''')
            
            if not has_pdf:
                print(f"[SKIP] No PDF button for: {title[:50]}")
                return False

            # Prepare filename
            safe_title = self.sanitize_filename(title)
            filename = f"{safe_title}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            
            # Handle duplicates
            counter = 1
            while os.path.exists(filepath):
                filename = f"{safe_title}_{counter}.pdf"
                filepath = os.path.join(self.download_dir, filename)
                counter += 1

            print(f"[DOWNLOAD] {title[:60]}...")

            # Set up download handler and click PDF button
            with page.expect_download(timeout=120000) as download_info:
                page.evaluate('''() => {
                    const btn = document.getElementById('pdf_save');
                    if (btn) {
                        btn.click();
                    }
                }''')

            download = download_info.value
            download.save_as(filepath)
            
            self.downloaded_docs.add(title)
            self.success_count += 1
            print(f"[SUCCESS] Saved: {filename}")
            return True

        except PlaywrightTimeout:
            print(f"[TIMEOUT] Download timed out: {title[:50]}")
            self.failed_downloads.append(title)
            return False
        except Exception as e:
            print(f"[ERROR] {title[:40]}: {str(e)[:50]}")
            self.failed_downloads.append(title)
            return False

    def go_back(self, page):
        """Navigate back using the ELEX back button"""
        try:
            page.click('button[title*="back"], #btn_back, .ebIcon-leftArrow')
            self.wait_for_load(page, 2)
        except:
            try:
                page.go_back()
                self.wait_for_load(page, 2)
            except:
                pass

    def process_category_recursively(self, page, depth=0):
        """Process all documents in current view and subcategories"""
        prefix = "  " * depth
        
        # Get documents in current table
        docs = self.get_document_links_from_table(page)
        
        if docs:
            print(f"{prefix}[INFO] Found {len(docs)} documents")
            
            for i, doc_title in enumerate(docs, 1):
                if doc_title in self.downloaded_docs:
                    continue
                    
                print(f"{prefix}[DOC {i}/{len(docs)}] {doc_title[:50]}")
                
                # Click document to open it
                if self.click_document_by_title(page, doc_title):
                    # Download PDF
                    self.download_current_pdf(page, doc_title)
                    # Go back to list
                    self.go_back(page)
                    self.wait_for_load(page, 1)

    def crawl_tree(self, page):
        """Main crawl logic - navigate tree and download all PDFs"""
        
        # Main categories to crawl
        main_categories = [
            "Library Overview",
            "Product Overview", 
            "Planning",
            "Operation and Maintenance",
            "Managed Object Management"
        ]
        
        # Sub-categories under Product Overview
        product_sub = [
            "System Overview",
            "System Optimization", 
            "Value Packages",
            "Features and Capacities",
            "Planned Features"
        ]
        
        # Sub-categories under Planning
        planning_sub = [
            "List Files",
            "RAN Information",
            "Solution Information"
        ]
        
        # Sub-categories under Operation and Maintenance
        oam_sub = [
            "Performance Management",
            "Software Management"
        ]
        
        print("\n[INFO] Starting tree crawl...")
        
        # First expand and process main categories
        for main_cat in main_categories:
            print(f"\n{'='*50}")
            print(f"[CATEGORY] {main_cat}")
            print('='*50)
            
            if self.expand_tree_node(page, main_cat):
                self.wait_for_load(page, 2)
                
                # Check for sub-categories
                if main_cat == "Product Overview":
                    for sub in product_sub:
                        print(f"\n  [SUB-CATEGORY] {sub}")
                        if self.expand_tree_node(page, sub):
                            self.wait_for_load(page, 2)
                            self.process_category_recursively(page, depth=2)
                            
                elif main_cat == "Planning":
                    for sub in planning_sub:
                        print(f"\n  [SUB-CATEGORY] {sub}")
                        if self.expand_tree_node(page, sub):
                            self.wait_for_load(page, 2)
                            self.process_category_recursively(page, depth=2)
                            
                elif main_cat == "Operation and Maintenance":
                    for sub in oam_sub:
                        print(f"\n  [SUB-CATEGORY] {sub}")
                        if self.expand_tree_node(page, sub):
                            self.wait_for_load(page, 2)
                            self.process_category_recursively(page, depth=2)
                else:
                    # Process main category directly
                    self.process_category_recursively(page, depth=1)

    def run(self):
        """Main crawler execution"""
        print(f"\n{'='*60}")
        print("ELEX PDF Crawler")
        print(f"{'='*60}")
        print(f"Library: {self.base_url}")
        print(f"Output: {self.download_dir}")
        print(f"{'='*60}\n")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=False,
                downloads_path=self.download_dir
            )
            
            context = browser.new_context(
                accept_downloads=True,
                viewport={"width": 1920, "height": 1080}
            )
            
            page = context.new_page()
            
            try:
                print("[INFO] Loading ELEX library...")
                page.goto(self.base_url, wait_until="networkidle", timeout=60000)
                self.wait_for_load(page, 5)
                
                print("[INFO] Starting crawl...")
                self.crawl_tree(page)
                
                # Print summary
                print(f"\n{'='*60}")
                print("DOWNLOAD COMPLETE")
                print(f"{'='*60}")
                print(f"Successfully downloaded: {self.success_count}")
                print(f"Failed: {len(self.failed_downloads)}")
                print(f"Files saved to: {self.download_dir}")
                
                if self.failed_downloads:
                    print(f"\nFailed documents ({len(self.failed_downloads)}):")
                    for title in self.failed_downloads[:20]:
                        print(f"  - {title[:60]}")
                    if len(self.failed_downloads) > 20:
                        print(f"  ... and {len(self.failed_downloads) - 20} more")
                
            except Exception as e:
                print(f"[FATAL] {e}")
                import traceback
                traceback.print_exc()
            finally:
                input("\nPress Enter to close browser...")
                browser.close()


def main():
    LIBRARY_URL = "http://localhost:9132/elex?db=LTE%20RAN%2020.Q1.1.alx"
    DOWNLOAD_DIR = "elex_downloads"
    
    crawler = ElexCrawler(LIBRARY_URL, DOWNLOAD_DIR)
    crawler.run()


if __name__ == "__main__":
    main()
