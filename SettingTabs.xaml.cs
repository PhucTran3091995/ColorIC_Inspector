using ColorIC_Inspector;
using System.Collections.ObjectModel;
using System.Windows;
using System.Windows.Controls;

namespace ColorIC_Inspector
{
    public partial class SettingTabs : UserControl
    {
        public ObservableCollection<ICType> ICTypes { get; set; }

        public SettingTabs()
        {
            InitializeComponent();

            // Mock Data for Settings
            ICTypes = new ObservableCollection<ICType> {
                new ICType { Name = "IC-74HC00", Count = 1, Color = "Black" },
                new ICType { Name = "IC-LM358", Count = 2, Color = "DarkGrey" }
            };
            lvICTypes.ItemsSource = ICTypes;
        }

        private void btnLogin_Click(object sender, RoutedEventArgs e)
        {
            if (txtPassword.Password == "admin")
            {
                pnlLogin.Visibility = Visibility.Collapsed;
                txtPassword.Password = "";
                txtLoginError.Text = "";
            }
            else
            {
                txtLoginError.Text = "Invalid Password";
            }
        }

        private void btnLogout_Click(object sender, RoutedEventArgs e)
        {
            pnlLogin.Visibility = Visibility.Visible;
        }

        private void btnBrowseModel_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.Filter = "YOLO Model (*.pt)|*.pt";
            if (dialog.ShowDialog() == true) txtModelPath.Text = dialog.FileName;
        }

        private void btnBrowseYaml_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.Filter = "YAML Config (*.yaml)|*.yaml";
            if (dialog.ShowDialog() == true) txtYamlPath.Text = dialog.FileName;
        }

        private void btnAddIC_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(txtNewICName.Text)) return;

            ICTypes.Add(new ICType
            {
                Name = txtNewICName.Text,
                Count = int.Parse(txtNewICCount.Text),
                Color = (cbNewICColor.SelectedItem as ComboBoxItem)?.Content.ToString()
            });

            txtNewICName.Text = "";
            txtNewICCount.Text = "1";
        }

        private void btnSave_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("Configuration Saved Successfully!", "System", MessageBoxButton.OK, MessageBoxImage.Information);
        }
    }
}